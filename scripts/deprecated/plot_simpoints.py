from gather_cluster_results import *
import os, sys
import plotly.graph_objects as go
import glob

color_list = [
# 35
"#696969",
"#8b4513",
"#006400",
"#808000",
"#ff0000",
"#483d8b",
"#daa520",
"#008080",
"#3cb371",
"#cd5c5c",
"#00008b",
"#32cd32",
"#800080",
"#d2b48c",
"#00ced1",
"#ff8c00",
"#00fa9a",
"#8a2be2",
"#adff2f",
"#dc143c",
"#00bfff",
"#0000ff",
"#da70d6",
"#b0c4de",
"#ff7f50",
"#00ff00",
"#ff00ff",
"#ffff54",
"#1e90ff",
"#ff1493",
"#7b68ee",
"#7fffd4",
"#f0e68c",
"#f0fff0",
"#ffc0cb",
]
color_list.reverse()

def read_all_for_stat(whole_sim_dir, num_of_dumps, file_prefix, s):
    # get all dump stat
    stats = []
    for dump in range(num_of_dumps):
        stats.append(get_acc_stat_from_file("{}/{}.period.{}".format(whole_sim_dir,
                                            file_prefix, dump), s.s_name, s.pos))

    # whole stat safe check
    # do not know the position of cumulative stat
    # assert(whole_stat == get_acc_stat_from_file("{}/{}.period.{}".format(whole_sim_dir,
    #                                             file_prefix, num_of_dumps), s.s_name, s.pos))

    return stats

def calculate_weighted_average_for_stat(points, stats):
    weighted_average = 0
    for point in points:
        weighted_average += point.weight * float(stats[point.seg_id])
    return weighted_average

def plot_for_stat(benchmark_name, simpoints, samples, labels, s, stats, whole_instruction_num, sample_weighted_instruction_num):

    stat_fig = go.Figure()

    # calculate whole stat
    whole_stat = sum(stats)

    # get simpoint stats as stars
    for simp in simpoints:
        stat_fig.add_trace(go.Scatter(x=[simp.seg_id],
                                        y=[stats[simp.seg_id]],
                                        mode = 'markers',
                                        name="cluster {}".format(simp.c_id),
                                        legendgroup=simp.c_id,
                                        showlegend=False,
                                        marker_color = color_list[simp.c_id],
                                        marker_line = dict(color='Black', width=1),
                                        marker_symbol = 'star',
                                        marker_size = 15))

    def add_cluster_trace(c_id, simp):
        print("adding {} with simp {}".format(c_id, simp if simp != None else "NA"))
        # seg_id
        seg_id_slice = [x.seg_id for x in labels if x.c_id == c_id]
        # stat
        stat_slice = [stats[seg_id] for seg_id in seg_id_slice]

        cluster_sum = sum(stat_slice)

        if simp != None:
            cluster_extrapolation = stats[simp.seg_id] * len(seg_id_slice)
            if whole_stat != 0:
                cluster_err = "{:.2%}".format((cluster_extrapolation - cluster_sum) / whole_stat)
            else:
                cluster_err = 0
            simp_stat = stats[simp.seg_id]
        else:
            cluster_extrapolation = "NA"
            cluster_err = "NA"
            simp_stat = "NA"

        stat_fig.add_trace(go.Bar(x = seg_id_slice, y = stat_slice,
                        name="cluster {}".format(c_id),
                        legendgroup=c_id,
                        hovertext="cluster size {}<br>simp stat {}<br>extrapolation {}<br>cluster sum {}<br>cluster error {}"
                                    .format(len(seg_id_slice),
                                            simp_stat,
                                            cluster_extrapolation,
                                            cluster_sum,
                                            cluster_err),
                        marker_color=color_list[c_id],
                        marker_line_color=color_list[c_id]))

    # the stat bars
    # seg_id and stat slices of the cluster
    for simp in simpoints:
        c_id = simp.c_id
        add_cluster_trace(c_id, simp)
        ####################################################################
        # # seg_id
        # seg_id_slice = [x.seg_id for x in labels if x.c_id == c_id]
        # # stat
        # stat_slice = [stats[seg_id] for seg_id in seg_id_slice]

        # cluster_extrapolation = stats[simp.seg_id] * len(seg_id_slice)
        # cluster_sum = sum(stat_slice)
        # cluster_err = (cluster_extrapolation - cluster_sum) / whole_stat
        # stat_fig.add_trace(go.Bar(x = seg_id_slice, y = stat_slice,
        #                 name="cluster {}".format(c_id),
        #                 legendgroup=c_id,
        #                 hovertext="cluster size {}<br>simp stat {}<br>extrapolation {}<br>cluster sum {}<br>cluster error {:.2%}"
        #                             .format(len(seg_id_slice),
        #                                     stats[simp.seg_id],
        #                                     cluster_extrapolation,
        #                                     cluster_sum,
        #                                     cluster_err),
        #                 marker_color=color_list[c_id],
        #                 marker_line_color=color_list[c_id]))
        ####################################################################

    # the simpoints could be filtered by a percentage.
    simpoints_clusters = set([simp.c_id for simp in simpoints])
    label_clusters = set([x.c_id for x in labels])
    assert simpoints_clusters.issubset(label_clusters)
    filtered_clusters = label_clusters.difference(simpoints_clusters)
    for c_id in filtered_clusters:
        add_cluster_trace(c_id, None)

    # get rates for the title
    weighted_average = calculate_weighted_average_for_stat(simpoints, stats)
    assert weighted_average == s.weighted_average, "{} != {}".format(weighted_average, s.weighted_average)

    # [0][0] is instructions, whose weighted_average shall be ready to use
    # to do: is this calculation accurate..?
    weighted_instruction_num = stat_groups[0].s_list[0].weighted_average
    simp_extrapolation = weighted_average * (whole_instruction_num / weighted_instruction_num)
    simp_err = ( simp_extrapolation / whole_stat - 1) if whole_stat != 0 else 0

    # sampling
    sample_weighted_average = calculate_weighted_average_for_stat(samples, stats)
    sample_extrapolation = sample_weighted_average * (whole_instruction_num / sample_weighted_instruction_num)
    sample_err = ( sample_extrapolation / whole_stat - 1) if whole_stat != 0 else 0

    # title
    y_max = max(stats)
    adjust = 0.05
    stat_fig.update_layout(
        title = "{}, {}, {}<br>warm_simpoint: {:.2f} ({:.2%}) v.s. warm_samples: {:.2f} ({:.2%})"
                .format(benchmark_name,
                        s.s_name,
                        whole_stat,
                        simp_extrapolation,
                        simp_err,
                        sample_extrapolation,
                        sample_err),
        xaxis_range = [0, len(labels)],
        yaxis_range = [0, y_max + y_max * adjust],
        bargap=0.0,
        legend_traceorder="grouped"
    )

    return stat_fig

def read_cluster_labels(sp_dir):
    labels = []
    with open(sp_dir + "/opt.l", "r") as f:
        for seg_id, line in enumerate(f):
            c_id = int(line.split()[0])
            # c_dis = int(line1.split()[1])
            labels.append(Simpoint(seg_id, 0, "not applicable", c_id))

    return labels

def get_num_of_dumps(whole_sim_dir):
    return len(glob.glob(whole_sim_dir + "/core.stat.0.out.period.*"))

def get_samples(simpoints, stats, whole_sim_dir):
    distance = int(len(stats) / (len(simpoints) + 1))

    sample_seg_ids = [i * distance for i in range(1, len(simpoints) + 1)]
    sample_inst_count = sum([stats[sample_seg_id] for sample_seg_id in sample_seg_ids])
    sample_weights = [stats[sample_seg_id] / sample_inst_count for sample_seg_id in sample_seg_ids]

    samples = []
    for c_id in range(0, len(sample_seg_ids)):
        samples.append(Simpoint(sample_seg_ids[c_id], sample_weights[c_id], whole_sim_dir, c_id))

    assert len(samples) == len(simpoints)
    assert samples[-1].seg_id < len(stats)

    sample_weighted_instruction_num = calculate_weighted_average_for_stat(samples, stats)
    assert abs(sample_inst_count / len(samples) - sample_weighted_instruction_num) < 1e-5, "{} and {}".format(sample_inst_count, sample_weighted_instruction_num)

    return samples, sample_weighted_instruction_num

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("expecting 3 parameters")
        exit()
    else:
        for parameter in sys.argv:
            print(parameter)

    BENCHNAME=sys.argv[1]
    SIMPOINTDIR=sys.argv[2]
    if not os.path.isdir(SIMPOINTDIR):
        print("simpoint directory {} does not exist!".format(SIMPOINTDIR))
        exit()
    WHOLESIMDIR=sys.argv[3]
    if not os.path.isdir(WHOLESIMDIR):
        print("whole simulation directory {} does not exist!".format(WHOLESIMDIR))
        exit()
    OUTDIR=sys.argv[4]
    if not os.path.isdir(OUTDIR):
        print("output directory {} does not exist!".format(OUTDIR))
        exit

    simpoints = read_simpoints(SIMPOINTDIR, WHOLESIMDIR, True)
    read_simpoint_stats(stat_groups, simpoints, True)
    # will calculate Stat.weighted_average, StatGroup.weighted_total, and Stat.weighted_ratio,
    calculate_weighted_average(stat_groups, simpoints)

    labels = read_cluster_labels(SIMPOINTDIR)
    num_of_dumps = get_num_of_dumps(WHOLESIMDIR)

    assert len(labels) == num_of_dumps

    for g_id, g in enumerate(stat_groups):
        print(g.g_name)
        for s_id, s in enumerate(g.s_list):
            stats = read_all_for_stat(WHOLESIMDIR, num_of_dumps, g.f_name, s)
            assert len(stats) == num_of_dumps
            if g_id == 0 and s_id == 0:
                assert g.g_name == "instructions", "the first group stat needs to be instructions"
                whole_instruction_num = sum(stats)
                samples, sample_weighted_instruction_num = get_samples(simpoints, stats, WHOLESIMDIR)
            fig = plot_for_stat(BENCHNAME, simpoints, samples, labels, s, stats, whole_instruction_num, sample_weighted_instruction_num)
            #  append to html file
            with open("{}/{}.html".format(OUTDIR, g.g_name), 'a') as f:
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))