#include "dr_api.h"
#include "drmgr.h"
#include "drutil.h"
#include "droption.h"
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <string>
#ifdef WINDOWS
# define DISPLAY_STRING(msg) dr_messagebox(msg)
#else
# define DISPLAY_STRING(msg) dr_printf("%s\n", msg)
#endif

using ::dynamorio::droption::DROPTION_SCOPE_CLIENT;
using ::dynamorio::droption::droption_t;

static droption_t<unsigned long long> op_segment_size
(DROPTION_SCOPE_CLIENT, "segment_size", 100000000/*100M*/, "specify the segment size",
 "Specify the size of the segments whose fingerprints will be collected. Default size is 100000000(100M).");

static droption_t<std::string> op_output
(DROPTION_SCOPE_CLIENT, "output", "bbfp", "specify the output file name",
 "Specify the output file name for both SimPoint and csv. Default name is bbfp.");

static droption_t<std::string> op_pcmap_output
(DROPTION_SCOPE_CLIENT, "pcmap_output", "pcmap", "specify the pcmap output file name",
 "Specify the output file name for pcmap. Default name is pcmap.");

static droption_t<std::string> op_footprint_output
(DROPTION_SCOPE_CLIENT, "footprint_output", "", "specify the ONLINE footprint output file name",
 "Specify the output file name for ONLINE instuction footprint. Default name is empty string, where footprint will not accumulate.");

static droption_t<bool> op_use_fetched_count
(DROPTION_SCOPE_CLIENT, "use_fetched_count", true, "specify if the segment size uses fetched count",
 "Fetched count considers a single fetch for rep string emulation. Default is true.");

static droption_t<bool> op_use_bb_pc
(DROPTION_SCOPE_CLIENT, "use_bb_pc", true, "specify if the fps use the bb pc as identifier",
 "Set to true to save the pc information. Default is true.");

typedef struct bb_counts {
    uint64 blocks;
    uint64 total_size;
    uint64 fetched_size;
    uint64 rep_string_count;
    void clear() {
        blocks = 0;
        total_size = 0;
        fetched_size = 0;
        rep_string_count = 0;
    }
} bb_counts;
// TODO: bb_id and as built blocks are the same?
// TODO: is iterating map producing the correct bb_id order?
enum Inspect_Cases {
  NON_EMULATION_NON_APP,
  NON_TRACE_NON_TRANSLATING_RECREATION,
  BIG_BB,
  NUM_INSPECTION
};

typedef struct per_thread_data {
    uint64 bb_id;
    uint64 thread_id;
    bb_counts counts_as_built;
    bb_counts counts_dynamic;
    std::map<uint64, uint64> fingerprints;
    std::map<uint64, std::vector<app_pc> > bb_pc_map;
    std::map<uint64, uint64> footprint;
    // search for pc_map, bb_pc_map
    uint64 num_of_segments;
    uint64 cur_counter;

    // track the previous bb executed to detect non-fetched rep emultaion
    uint64 prev_first_addr;

    std::map<app_pc, std::pair<uint64, uint>> addr_to_bb_for_trace;

    uint64 inspect_case_as_built[NUM_INSPECTION];

    void clear() {
        bb_id = 0;
        thread_id = 0;
        counts_as_built.clear();
        counts_dynamic.clear();
        fingerprints.clear();
        num_of_segments = 0;
        cur_counter = 0;
        prev_first_addr = 0;
        addr_to_bb_for_trace.clear();
        memset(inspect_case_as_built, 0, sizeof(inspect_case_as_built));
    }
} per_thread_data;

typedef struct pc_marker {
   app_pc pc;
   uint64 freq;
    pc_marker(app_pc pc_val, uint64 freq_val) {
        pc = pc_val;
        freq = freq_val;
    }
} pc_marker;

/* Cross-instrumentation-phase data. */
// typedef struct {
//     uint64 bb_id;
// } instru_data_t;

int tls_idx;

void *as_built_lock;
void *count_lock;

static uint64 per_instr_count;
static uint64 per_cur_counter;

static std::vector<pc_marker> pc_markers;

static bool global_is_emulation = false;

static void
event_exit(void);
static void
fork_init(void *drcontext);
static void
event_thread_init(void *drcontext);
static void
event_thread_exit(void *drcontext);

static dr_emit_flags_t
event_bb_app2app(void *drcontext, void *tag, instrlist_t *bb, bool for_trace,
                 bool translating);

static dr_emit_flags_t
event_app_analysis(void *drcontext, void *tag, instrlist_t *bb,
                   bool for_trace, bool translating, OUT void **user_data);

static dr_emit_flags_t
event_app_instruction(void *drcontext, void *tag, instrlist_t *bb,
                       instr_t *inst, bool for_trace, bool translating, void *user_data);

static void
clean_call(uint instruction_count, uint64 bb_id, uint64 segment_size, uint emulation_start_count, uint64 first_addr, uint is_rep_emulation);

// identical to memtrace post processing
// excepty for assertions
uint64_t output_fingerprint(std::string file_name, std::map<uint64_t, uint64_t> fingerprint, bool exit);

DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{
    using ::dynamorio::droption::droption_parser_t;
    std::string parse_err;
    int last_index;
    if (!droption_parser_t::parse_argv(DROPTION_SCOPE_CLIENT, argc, argv, &parse_err, &last_index)) {
        dr_printf( "Usage error: %s \n at: %d%", parse_err.c_str(), last_index);
        dr_abort();
    }
    dr_printf("The segment size: %llu\n", op_segment_size.get_value());
    dr_printf("The output prefix: %s\n", op_output.get_value().c_str());
    dr_printf("The foorprint output prefix: %s\n", op_footprint_output.get_value().c_str());
    dr_printf("The pcmap output prefix: %s\n", op_pcmap_output.get_value().c_str());
    dr_printf("op_use_fetched_count: %d\n", op_use_fetched_count.get_value());
    dr_printf("op_use_bb_pc: %d\n", op_use_bb_pc.get_value());

    if (!drmgr_init() || !drutil_init())
        DR_ASSERT(false);

    /* register events */
    dr_register_exit_event(event_exit);

    dr_register_fork_init_event(fork_init);

    // will null priority be problematic?
    if (!drmgr_register_thread_init_event(event_thread_init) ||
        !drmgr_register_thread_exit_event_ex(event_thread_exit, NULL))
        DR_ASSERT(false);

    // dr_register_bb_event(event_basic_block);
    if (!drmgr_register_bb_app2app_event(event_bb_app2app, NULL))
        DR_ASSERT(false);
    if (!drmgr_register_bb_instrumentation_event(event_app_analysis, /*event_app_instruction*/ NULL, NULL))
        DR_ASSERT(false);

    /* initialize lock */
    as_built_lock = dr_mutex_create();
    count_lock = dr_mutex_create();

    tls_idx = drmgr_register_tls_field();
}

static void
event_exit(void)
{
    /* Display results - we must first snpritnf the string as on windows
     * dr_printf(), dr_messagebox() and dr_fprintf() can't print floats. */
    char msg[512];
    int len;
    len = snprintf(msg, sizeof(msg)/sizeof(msg[0]),
                    "Number of total instructions as per: %"UINT64_FORMAT_CODE"\n",
                    per_instr_count);
    DR_ASSERT(len > 0);
    msg[sizeof(msg)/sizeof(msg[0])-1] = '\0';
    DISPLAY_STRING(msg);

    // assert(per_instr_count == counts_dynamic.total_size);

    // if (!drmgr_unregister_tls_field(tls_idx) ||
    //     !drmgr_unregister_thread_init_event(event_thread_init) ||
    //     !drmgr_unregister_thread_exit_event(event_thread_exit))
    //     DR_ASSERT(false);

    /* free mutex */
    dr_mutex_destroy(as_built_lock);
    dr_mutex_destroy(count_lock);
}

static void
fork_init(void *drcontext)
{
    // per thread data get copied
    // dunno why that dr_fprintf(STDERR, ) does not work after fork
    per_thread_data *t_data = (per_thread_data *) drmgr_get_tls_field(drcontext, tls_idx);
    t_data->clear();
    t_data->thread_id = dr_get_thread_id(drcontext);
    dr_printf("[%llu] fork init\n", t_data->thread_id);
}

static void
event_thread_init(void *drcontext)
{
    /* create an instance of our data structure for this thread */
    per_thread_data *t_data = (per_thread_data *)dr_thread_alloc(drcontext, sizeof(per_thread_data));
    *t_data = {};
    t_data->clear();
    t_data->thread_id = dr_get_thread_id(drcontext);
    dr_printf("[%llu] new thread\n", t_data->thread_id);

    /* store it in the slot provided in the drcontext */
    drmgr_set_tls_field(drcontext, tls_idx, t_data);
}

static void
event_thread_exit(void *drcontext)
{
    per_thread_data *t_data = (per_thread_data *) drmgr_get_tls_field(drcontext, tls_idx);
    DR_ASSERT(t_data->thread_id == dr_get_thread_id(drcontext));

    dr_printf("[%llu] exit\n", t_data->thread_id);

    /* NOTE - if we so choose we could report per-thread sizes here. */
    /* for now I'll just use lock*/
    // dr_mutex_lock(count_lock);
    // counts_dynamic.blocks += counts->blocks;
    // counts_dynamic.total_size += counts->total_size;
    // dr_mutex_unlock(count_lock);

    // mycsv.open (op_output.get_value() + "." + std::to_string(dr_get_thread_id(drcontext)) + ".csv");
    // mymarker.open(op_output.get_value() + ".mk");
    // T:1540:4   :1541:2

    // bool witness_total = false;

    // WARN: actually if counts_dynamic.total_size % seg size == 0,
    // fingerprints would be one size greater than pc_markers.size()
    // rare but then the implementation would be off
    // DR_ASSERT(fingerprints.size() == pc_markers.size());

    // DR_ASSERT(fingerprints.find(t_data->thread_id) != fingerprints.end());

    std::map<uint64, uint64>& fp = t_data->fingerprints;
    if (fp.size() > 0) {
        t_data->num_of_segments++;

        uint64_t instrs_count = output_fingerprint(op_output.get_value() + "." + std::to_string(dr_get_thread_id(drcontext)), fp, true);
        // can move the following two into the func; but need more parameters
        dr_printf("[%llu] exiting, num of instrs within last segment: %llu\n", dr_get_thread_id(drcontext), instrs_count);
        if (!op_use_fetched_count.get_value()) {
            DR_ASSERT(instrs_count == t_data->counts_dynamic.total_size % op_segment_size.get_value());
        }
        if (op_footprint_output.get_value().size()) {
            DR_ASSERT(t_data->footprint.size());
            uint64_t instrs_count_footprint = output_fingerprint(op_footprint_output.get_value() + "." + std::to_string(dr_get_thread_id(drcontext)), t_data->footprint, true);
            DR_ASSERT(instrs_count_footprint == instrs_count);
        }
    }

    // DR_ASSERT(witness_total);

    // snprintf will seg fault?
    dr_printf(
                    "========================================================\n"
                    "Thread: %lld\n"
                    "Number of blocks built : %"UINT64_FORMAT_CODE"\n"
                    "     Average size      : %5.2lf instructions\n"
                    "Number of blocks executed  : %"UINT64_FORMAT_CODE"\n"
                    "     Average weighted size : %5.2lf instructions\n"
                    "Number of total instructions   : %"UINT64_FORMAT_CODE"\n"
                    "Number of fetched instructions : %"UINT64_FORMAT_CODE"\n"
                    "     as-built rep    : %"UINT64_FORMAT_CODE"\n"
                    "     dynamic rep     : %"UINT64_FORMAT_CODE"\n"
                    // "Number of total instructions as per: %"UINT64_FORMAT_CODE"\n"
                    "Number of inspected case:\n"
                    "     non_emulation_non_app                : %"UINT64_FORMAT_CODE"\n"
                    "     non_trace_non_translating_recreation : %"UINT64_FORMAT_CODE"\n"
                    "     big_bb                               : %"UINT64_FORMAT_CODE"\n"
                    "========================================================\n"
                    ,
                    t_data->thread_id,
                    t_data->counts_as_built.blocks,
                    t_data->counts_as_built.total_size / (double)t_data->counts_as_built.blocks,
                    t_data->counts_dynamic.blocks,
                    t_data->counts_dynamic.total_size / (double)t_data->counts_dynamic.blocks,
                    t_data->counts_dynamic.total_size,
                    t_data->counts_dynamic.fetched_size,
                    t_data->counts_as_built.rep_string_count,
                    t_data->counts_dynamic.rep_string_count,
                    // per_instr_count,
                    t_data->inspect_case_as_built[NON_EMULATION_NON_APP],
                    t_data->inspect_case_as_built[NON_TRACE_NON_TRANSLATING_RECREATION],
                    t_data->inspect_case_as_built[BIG_BB]
                    );

    // dumping bb pc map
    std::ofstream mypcmap;
    mypcmap.open(op_pcmap_output.get_value() + "." + std::to_string(dr_get_thread_id(drcontext)), std::ofstream::out);
    if (!mypcmap.is_open()) {
        dr_printf("open pcmap file failed\n");
    }
    std::cout <<"dumping pc map, total unique bb: " << t_data->addr_to_bb_for_trace.size() << std::endl;
    DR_ASSERT(t_data->addr_to_bb_for_trace.size() == t_data->bb_pc_map.size());
    mypcmap << "bb_pc,bb_id,bb_size,bb_pc_vec" << std::endl;
    // pc -> bb_id, bb_size
    std::map<app_pc, std::pair<uint64, uint>>::iterator bb_info;
    uint64 unique_instrs_count = 0;
    for (bb_info = t_data->addr_to_bb_for_trace.begin(); bb_info != t_data->addr_to_bb_for_trace.end(); bb_info++) {
        unique_instrs_count += bb_info->second.second;
        mypcmap << (uint64)bb_info->first << "," << bb_info->second.first << "," << bb_info->second.second << ",";
        // using bb_pc_map for complete bb pc vector!

        DR_ASSERT(t_data->bb_pc_map.find(bb_info->second.first) != t_data->bb_pc_map.end());
        const std::vector<app_pc> & bb_pc = t_data->bb_pc_map[bb_info->second.first];
        DR_ASSERT(bb_info->second.second == bb_pc.size());
        DR_ASSERT(bb_info->first == bb_pc.front());
        for (app_pc pc : bb_pc) {
            mypcmap << (uint64)pc << "-";
        }
        mypcmap << std::endl;
    }
    std::cout <<"total unique instrs: " << unique_instrs_count << std::endl;
    mypcmap.close();

    dr_thread_free(drcontext, t_data, sizeof(per_thread_data));

    dr_printf("[-] exited\n");
}

static void
clean_call(uint instruction_count, uint64 bb_id, uint64 segment_size, uint emulation_start_count, uint64 first_addr, uint is_rep_emulation)
{
    void *drcontext = dr_get_current_drcontext();
    per_thread_data *t_data = (per_thread_data *) drmgr_get_tls_field(drcontext, tls_idx);
    DR_ASSERT(t_data->thread_id == dr_get_thread_id(drcontext));

    // increment inst counter
    // increment PC map
    // push the marker
    // the (to_new_vector_count + 1)th pc
    std::vector<app_pc> bb_pc;
    if (op_footprint_output.get_value().size()) {
        bb_pc = t_data->bb_pc_map[bb_id];
    }

    // for (uint i = 0; i < bb_pc.size(); i++) {
        // app_pc cur_pc = bb_pc[i];
        // t_data->footprint[(uint64)cur_pc]++;
        // if (cur_counter % segment_size == 1) {
        //     pc_markers.push_back(pc_marker(cur_pc, t_data->footprint[cur_pc]));
        //     std::cout<<"marker pushed: " << (void *)cur_pc << ", " << t_data->footprint[cur_pc] << std::endl;
        //     dr_printf("marker pushed: %ld, %ld\n", cur_pc, t_data->footprint[cur_pc]);
        //     dr_printf("marker pushed: %ld, %ld\n", (void *)cur_pc, t_data->footprint[cur_pc]);
        //     dr_printf("marker pushed: %p, %ld\n", cur_pc, t_data->footprint[cur_pc]);
        //     dr_printf("marker pushed: %p, %ld\n", (void *)cur_pc, t_data->footprint[cur_pc]);
        // }
    // }

    if (is_rep_emulation && first_addr == t_data->prev_first_addr) {
        // We are not counting the REP consequent instructions for basic block vectors
        return;
    } else {
        // otherwise just increment
        if (is_rep_emulation) {
            DR_ASSERT(instruction_count == 1);
        }
        t_data->cur_counter += instruction_count;
        t_data->counts_dynamic.fetched_size += instruction_count;
    }

    t_data->prev_first_addr = first_addr;
    t_data->counts_dynamic.rep_string_count += emulation_start_count;
    t_data->counts_dynamic.blocks++;
    t_data->counts_dynamic.total_size += instruction_count;

    uint to_last_vector_count = 0;
    uint to_new_vector_count = 0;

    // if at a boundary (excluding perfect aligned boundary)
    if (t_data->cur_counter > segment_size) {
        // rep emulation bb cannot make it here
        // no matter fetched or not
        DR_ASSERT(!is_rep_emulation);
        // so the calculation can assume that
        // all insts in this bb are fetched
        // and op_use_fetched_count does not matter
        to_new_vector_count = t_data->cur_counter - segment_size;
        to_last_vector_count = instruction_count - to_new_vector_count;
    } else {
        to_last_vector_count = instruction_count;
    }

    DR_ASSERT(to_last_vector_count + to_new_vector_count == instruction_count);
    DR_ASSERT((t_data->cur_counter > segment_size) == (to_new_vector_count > 0));

    if (op_footprint_output.get_value().size()) {
        for (uint i = 0; i < to_last_vector_count; i++) {
            app_pc cur_pc = bb_pc[i];
            t_data->footprint[(uint64)cur_pc]++;
        }
    }
    std::map<uint64, uint64>& fp = t_data->fingerprints;

    // or just fp[bb_id] += instruction_count;
    if (op_use_bb_pc.get_value()) {
        fp[first_addr] += to_last_vector_count;
    } else {
        fp[bb_id] += to_last_vector_count;
    }

    // if (fp.find(bb_id) == fp.end()) {
    //     fp.insert(std::make_pair(bb_id, 0));
    //     // std::cout << "new bb detected: " << bb_id << std::endl;
    //     dr_printf("[%d] detected bb_id %d\n", t_data->thread_id, bb_id);
    // }
    // // weight each bb by its size
    // fp.at(bb_id) += to_last_vector_count;

    // no weight each bb
    // fp.at(bb_id) += 1;

    // including perfect alignment
    if (t_data->cur_counter >= segment_size) {
        t_data->num_of_segments++;
        dr_printf("[%llu] to be appened segment num %llu\n", t_data->thread_id, t_data->num_of_segments);

        t_data->cur_counter = 0;

        uint64_t instrs_count = output_fingerprint(op_output.get_value() + "." + std::to_string(dr_get_thread_id(drcontext)), fp, false);
        if (op_footprint_output.get_value().size()) {
            uint64_t instrs_count_footprint = output_fingerprint(op_footprint_output.get_value() + "." + std::to_string(dr_get_thread_id(drcontext)), t_data->footprint, false);
            DR_ASSERT(instrs_count_footprint == instrs_count);
            t_data->footprint.clear();
        }
        // clear for the next segment
        fp.clear();

        // record the residue
        if (to_new_vector_count > 0) {
            if (op_footprint_output.get_value().size()) {
                for (uint i = to_last_vector_count; i < instruction_count; i++) {
                    app_pc cur_pc = bb_pc[i];
                    t_data->footprint[(uint64)cur_pc]++;
                }
            }
            if (op_use_bb_pc.get_value()) {
                fp.insert(std::make_pair(first_addr, to_new_vector_count));
            } else {
                fp.insert(std::make_pair(bb_id, to_new_vector_count));
            }
            t_data->cur_counter = to_new_vector_count;
        }
    }
    // if (cur_counter == 0)
    //     DR_ASSERT(fingerprints.size() == pc_markers.size() + 1);
    // else
    //     DR_ASSERT(fingerprints.size() == pc_markers.size());
}

static void
clean_call_for_all_instr(/*uint64 bb_id, */uint64 segment_size, app_pc addr)
{
    dr_printf("line: %d\n", __LINE__);

    dr_mutex_lock(count_lock);

    per_instr_count++;
    per_cur_counter++;

    // 1. this clean call is not used as per inst instrumentation did not work for me
    // 2. pc_map ~ is replaced by per-thread footprint
    // pc_map[addr]++;

    if (per_cur_counter == segment_size) {
        per_cur_counter = 0;
    }

    // if at the start
    if (per_cur_counter == 1) {
        // pc_markers.push_back(pc_marker(addr, pc_map[addr]));
        std::cout<< "cur markers size: " << pc_markers.size() << std::endl;
    }

    dr_mutex_unlock(count_lock);
}

/* We transform string loops into regular loops so we can more easily
 * monitor every memory reference they make.
 */
static dr_emit_flags_t
event_bb_app2app(void *drcontext, void *tag, instrlist_t *bb, bool for_trace,
                 bool translating)
{
    /* drbbdup doesn't pass the user_data from this stage so we use TLS.
     * XXX i#5400: Integrating drbbdup into drmgr would provide user_data here.
     */
    if (!drutil_expand_rep_string(drcontext, bb)) {
        DR_ASSERT(false);
        /* in release build, carry on: we'll just miss per-iter refs */
    }
    /*if (!drx_expand_scatter_gather(drcontext, bb, &pt->scatter_gather)) {
        DR_ASSERT(false);
    } */
    return DR_EMIT_DEFAULT;
}

/* copied from dynamorio clients/drcachesim/tracer/instru.cpp */
int
count_app_instrs(instrlist_t *ilist, bool for_trace = false)
{
    static int bb_built_count = 0;
    int count = 0;
    bool in_emulation_region = false;
    bool been_in_emulation = false;
    for (instr_t *inst = instrlist_first(ilist); inst != NULL;
         inst = instr_get_next(inst)) {
        if (!in_emulation_region && drmgr_is_emulation_start(inst)) {
            in_emulation_region = true;
            // Each emulation region corresponds to a single app instr.
            ++count;
            // if(!for_trace)
            //     dr_fprintf(STDERR, "[%d]: ad: %p, op: %d\n", bb_built_count, (void *)instr_get_app_pc(inst), instr_get_opcode(inst));
        }
        if (!in_emulation_region && instr_is_app(inst)) {
            // Hooked native functions end up with an artificial jump whose translation
            // is its target.  We do not want to count these.
            if (!(instr_is_ubr(inst) && opnd_is_pc(instr_get_target(inst)) &&
                  opnd_get_pc(instr_get_target(inst)) == instr_get_app_pc(inst))) {
                ++count;
                if (been_in_emulation) {
                    // if(!for_trace)
                    //     dr_fprintf(STDERR, "[ae]");
                }
                // if(!for_trace)
                //     dr_fprintf(STDERR, "[%d]: ad: 0x%p, op: %d\n", bb_built_count, (void *)instr_get_app_pc(inst), instr_get_opcode(inst));

            }
        }
        if (in_emulation_region && drmgr_is_emulation_end(inst)) {
            in_emulation_region = false;
            if(!for_trace)
                been_in_emulation = true;
        }
    }
    if(!for_trace)
        bb_built_count++;
    return count;
}

// ref: ext/drutil/drutil.c
bool copy_opc_is_stringop_loop(uint opc)
{
    return (opc == OP_rep_ins || opc == OP_rep_outs || opc == OP_rep_movs ||
            opc == OP_rep_stos || opc == OP_rep_lods || opc == OP_rep_cmps ||
            opc == OP_repne_cmps || opc == OP_rep_scas || opc == OP_repne_scas);
}

/* samples: inscount and memtrace_x86
 *
 * */
static dr_emit_flags_t
event_app_analysis(void *drcontext, void *tag, instrlist_t *bb,
                  bool for_trace, bool translating, OUT void **user_data)
{
    // dr_printf("bb tag: %p\n", tag);

    uint num_instrs = 0;
    uint emulation_start_count = 0;
    uint non_emulation_non_app = 0;
    instr_t *instr;

    per_thread_data *t_data = (per_thread_data *) drmgr_get_tls_field(drcontext, tls_idx);
    // DR_ASSERT(fingerprints.find(t_data->thread_id) != fingerprints.end());
    DR_ASSERT(t_data->thread_id == dr_get_thread_id(drcontext));

    /* count the number of instructions in this block */
    // for (instr = instrlist_first(bb); instr != NULL; instr = instr_get_next(instr)) {
    //     num_instructions++;
    // }

    bool is_emulation = false;
    uint emulation_length = 0;
    bool is_rep_emulation = false;
    std::vector<app_pc> bb_pc;
    bool first = true;
    app_pc first_addr;
    for (instr = instrlist_first(bb); instr != NULL; instr = instr_get_next(instr)) {
        if (drmgr_is_emulation_start(instr)) {
            /* Each emulated instruction is replaced by a series of native
             * instructions delimited by labels indicating when the emulation
             * sequence begins and ends. It is the responsibility of the
             * emulation client to place the start/stop labels correctly.
             */
            DR_ASSERT(instr == instrlist_first(bb));
            num_instrs++;

            DR_ASSERT(first);
            first = false;
            // instr_t *instr_fetch = drmgr_orig_app_instr_for_fetch(drcontext);
            // DR_ASSERT(instr_fetch);
            first_addr = instr_get_app_pc(instr);
            DR_ASSERT(!first_addr);

            // get real app pc hecky hecky way
            emulated_instr_t emulation_info;
            bool ok = drmgr_get_emulated_instr_data(instr, &emulation_info);
            DR_ASSERT(ok);
            first_addr = instr_get_app_pc(emulation_info.instr);
            DR_ASSERT(first_addr);
            int rep_op = instr_get_opcode(emulation_info.instr);
            DR_ASSERT(copy_opc_is_stringop_loop(rep_op));
            is_rep_emulation = true;
            bb_pc.push_back(first_addr);

            emulation_start_count++;
            is_emulation = true;
            /* Data about the emulated instruction can be extracted from the
             * start label using the accessor function:
             * drmgr_get_emulated_instr_data()
             */
            continue;
        }
        if (drmgr_is_emulation_end(instr)) {
            // in our case the rep expansion has no end label
            DR_ASSERT(0);
            DR_ASSERT(instr == instrlist_last(bb));
            is_emulation = false;
            // std::cout<< "end emulation seq length: " << emulation_length << std::endl;
            continue;
        }
        if (is_emulation) {
            emulation_length++;
            if (instr_get_next(instr) == NULL) {
                std::cout<< "inc emulation till end: " << emulation_length << std::endl;
            }
            continue;
        }
        if (!instr_is_app(instr)) {
            non_emulation_non_app++;
            continue;
        }
        bb_pc.push_back(instr_get_app_pc(instr));
        num_instrs++;
        if (first) {
            first = false;
            first_addr = instr_get_app_pc(instr);
        }
    }

    DR_ASSERT(first_addr);
    int compare_count = count_app_instrs(bb, for_trace);

    DR_ASSERT(first_addr == bb_pc.front());
    DR_ASSERT(num_instrs == bb_pc.size());
    // for (int ii = 0; ii < bb_pc.size(); ii++) {
    //     dr_printf("%ld pc: %ld", counts_as_built.blocks+1, bb_pc[ii]);
    // }
    /* update the as-built counts */
    dr_mutex_lock(as_built_lock);
    if (emulation_start_count) {
        std::cout<< "emulation_start_count: " <<  emulation_start_count << std::endl;
        DR_ASSERT(emulation_start_count == 1);
        DR_ASSERT(num_instrs == 1);
    }
    dr_mutex_unlock(as_built_lock);

    // thread private no need lock
    uint64 cleancall_bb_id;
    // a basic block can be created multiple times;
    // see https://groups.google.com/g/dynamorio-users/c/DDHfM_mB9Vg/m/NxIjLfLjAQAJ?pli=1

    if (t_data->addr_to_bb_for_trace.find(first_addr) == t_data->addr_to_bb_for_trace.end()) {
        DR_ASSERT(!for_trace);

        t_data->counts_as_built.rep_string_count += emulation_start_count;
        t_data->counts_as_built.blocks++;
        t_data->counts_as_built.total_size += num_instrs;
        t_data->counts_as_built.fetched_size += num_instrs;
        t_data->bb_id++;

        cleancall_bb_id = t_data->bb_id;
        dr_printf("[%llu] a new bb_id %llu\n", dr_get_thread_id(drcontext), cleancall_bb_id);

        // t_data->addr_to_bb_for_trace.insert(std::make_pair(first_addr, std::make_pair(cleancall_bb_id, num_instrs)));
        t_data->addr_to_bb_for_trace[first_addr] = std::make_pair(cleancall_bb_id, num_instrs);
        DR_ASSERT(cleancall_bb_id == t_data->addr_to_bb_for_trace.size());

        DR_ASSERT(t_data->bb_pc_map.find(cleancall_bb_id) == t_data->bb_pc_map.end());
        t_data->bb_pc_map[cleancall_bb_id] = bb_pc;

        // inspections
        t_data->inspect_case_as_built[NON_EMULATION_NON_APP] += non_emulation_non_app;
        if (num_instrs > 2048) {
            t_data->inspect_case_as_built[BIG_BB]++;
        }
    } else {
        // get the bb id for the bb in trace
        std::map<app_pc, std::pair<uint64, uint>>::iterator bb_info = t_data->addr_to_bb_for_trace.find(first_addr);
        cleancall_bb_id = bb_info->second.first;
        dr_printf("[%llu] re-create bb_id %llu, for_trace: %d, translating: %d\n", dr_get_thread_id(drcontext), cleancall_bb_id, for_trace, translating);

        if (num_instrs != bb_info->second.second) {
            dr_printf("[%llu] re-created bb_id %llu has different inst count: %u v.s. %u.\n", dr_get_thread_id(drcontext), cleancall_bb_id, num_instrs, bb_info->second.second);
        }
        DR_ASSERT(num_instrs == bb_info->second.second);

        DR_ASSERT(t_data->bb_pc_map.find(cleancall_bb_id) != t_data->bb_pc_map.end());
        const std::vector<app_pc> & existed_bb_pc = t_data->bb_pc_map[cleancall_bb_id];
        if (bb_pc != existed_bb_pc) {
            dr_printf("[%llu] re-created bb_id %llu has different bb content; bb size: %lu v.s. %lu.\n", dr_get_thread_id(drcontext), cleancall_bb_id, bb_pc.size(), existed_bb_pc.size());
            dr_printf("re-created bb:\n");
            for (app_pc pc : bb_pc) {
                dr_printf("%llu\n", (uint64)pc);
            }
            dr_printf("existed bb:\n");
            for (app_pc pc : existed_bb_pc) {
                dr_printf("%llu\n", (uint64)pc);
            }
        }
        DR_ASSERT(bb_pc == existed_bb_pc);

        // inspections
        if (!for_trace && !translating) {
            t_data->inspect_case_as_built[NON_TRACE_NON_TRANSLATING_RECREATION]++;
        }
    }

    if (compare_count != num_instrs) {
        dr_printf("[%llu] bb_id %llu has different inst count %d %u from function\n", dr_get_thread_id(drcontext), cleancall_bb_id, compare_count, num_instrs);
    }
    DR_ASSERT(compare_count == num_instrs);

    /* insert clean call */
    dr_insert_clean_call(drcontext, bb, instrlist_first(bb), (void *)clean_call, false, 6,
                        OPND_CREATE_INT32(num_instrs), OPND_CREATE_INT64(cleancall_bb_id),
                        OPND_CREATE_INT64(op_segment_size.get_value()),
                        OPND_CREATE_INT32(emulation_start_count),
                        OPND_CREATE_INT64(first_addr),
                        OPND_CREATE_INT32(is_rep_emulation)
                        );

    return DR_EMIT_DEFAULT;
}

static dr_emit_flags_t
event_app_instruction(void *drcontext, void *tag, instrlist_t *bb,
                       instr_t *inst, bool for_trace, bool translating, void *user_data)
{
    dr_printf("line: %d\n", __LINE__);
    drmgr_disable_auto_predication(drcontext, bb);

    // 1.
    instr_t * real_inst = drmgr_orig_app_instr_for_fetch(drcontext);

    // if (real_inst != NULL) {

    //     dr_insert_clean_call(drcontext, bb, real_inst, (void *)clean_call_for_all_instr,
    //                         false, 2, /*OPND_CREATE_INT64(bb_id),*/
    //                         OPND_CREATE_INT64(op_segment_size.get_value()),
    //                         OPND_CREATE_INTPTR(instr_get_app_pc(real_inst)));
    // }

    // 2.
    // dr_insert_clean_call(drcontext, bb, inst, (void *)clean_call_for_all_instr,
    //                     false, 2, /*OPND_CREATE_INT64(bb_id),*/
    //                     OPND_CREATE_INT64(op_segment_size.get_value()),
    //                     OPND_CREATE_INTPTR(instr_get_app_pc(inst)));

    // 3.
    uint emulation_length = 0;
    if (drmgr_is_emulation_start(inst)) {
        // assert(inst == instrlist_first(bb));
        global_is_emulation = true;
    dr_printf("line: %d\n", __LINE__);

        if (real_inst != NULL) {
            dr_insert_clean_call(drcontext, bb, real_inst, (void *)clean_call_for_all_instr,
                                false, 2, /*OPND_CREATE_INT64(bb_id),*/
                                OPND_CREATE_INT64(op_segment_size.get_value()),
                                OPND_CREATE_INTPTR(instr_get_app_pc(real_inst)));
        }

        // dr_insert_clean_call(drcontext, bb, inst, (void *)clean_call_for_all_instr,
        //                     false, 2, /*OPND_CREATE_INT64(bb_id),*/
        //                     OPND_CREATE_INT64(op_segment_size.get_value()),
        //                     OPND_CREATE_INTPTR(instr_get_app_pc(inst)));
    dr_printf("line: %d\n", __LINE__);

        return DR_EMIT_DEFAULT;
    }
    if (drmgr_is_emulation_end(inst)) {
        global_is_emulation = false;
        // std::cout<< "end emulation seq length: " << emulation_length << std::endl;
    dr_printf("line: %d\n", __LINE__);

        return DR_EMIT_DEFAULT;
    }
    if (global_is_emulation) {
        emulation_length++;
        if (instr_get_next(inst) == NULL) {
            std::cout<< "instrument emulation till end: " << emulation_length << std::endl;
            global_is_emulation = false;
        }
    dr_printf("line: %d\n", __LINE__);

        return DR_EMIT_DEFAULT;
    }
    dr_printf("line: %d\n", __LINE__);

    if (real_inst != NULL) {

        dr_insert_clean_call(drcontext, bb, real_inst, (void *)clean_call_for_all_instr,
                            false, 2, /*OPND_CREATE_INT64(bb_id),*/
                            OPND_CREATE_INT64(op_segment_size.get_value()),
                            OPND_CREATE_INTPTR(instr_get_app_pc(real_inst)));
    }
    // dr_insert_clean_call(drcontext, bb, inst, (void *)clean_call_for_all_instr,
    //                     false, 2, /*OPND_CREATE_INT64(bb_id),*/
    //                     OPND_CREATE_INT64(op_segment_size.get_value()),
    //                     OPND_CREATE_INTPTR(instr_get_app_pc(inst)));
    dr_printf("line: %d\n", __LINE__);

    return DR_EMIT_DEFAULT;
}

// is also used to print footprint
uint64_t output_fingerprint(std::string file_name, std::map<uint64_t, uint64_t> fingerprint, bool exit) {
    // output the map for this segment
    // make it a function?
    std::ofstream myfile;
    myfile.open(file_name, std::ofstream::out | std::ofstream::app);

    if (!myfile.is_open()) {
        std::cout << "open file failed: " << file_name << std::endl;
    }

    // std::cout << num_of_segments << "th fp dimensions: " << fingerprint.size() << std::endl;
    // fine if comment starting here
    std::map<uint64_t, uint64_t>::iterator freq;
    uint64_t instrs_count = 0;

    uint64_t nonzero_count = 0;
    // static std::vector<uint64> csv_line(counts_as_built.blocks, 0);

    for (freq = fingerprint.begin(); freq != fingerprint.end(); freq++) {
        instrs_count += freq->second;
        if(freq == fingerprint.begin()) {
            myfile << "T";
        }
        myfile << ":" << freq->first << ":" << freq->second << " ";

        // csv_line[freq->first] = freq->second;
        nonzero_count++;

        // if (freq->first + 1 == counts_as_built.blocks) {
        //     witness_total = true;
        // }
    }

    DR_ASSERT(nonzero_count == fingerprint.size());
    if (!exit) {
        if (!op_use_fetched_count.get_value()) {
            DR_ASSERT(instrs_count == op_segment_size.get_value());
        }
    }

    myfile << std::endl;
    myfile.close();

    return instrs_count;
}