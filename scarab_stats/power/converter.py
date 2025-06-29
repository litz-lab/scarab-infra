#!/usr/bin/env python3

# Copyright (c) 2025 University of California, Santa Cruz. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ----------------------------------------------------------------------------
#
# File: converter.py
# Author: Yinyuan Zhao (Litz Lab)
#
# ----------------------------------------------------------------------------

import os
import json
import math
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from collections import OrderedDict


class Converter:
    """
    A class to convert between McPAT-style XML and structured JSON representation.
    Exposed API:
      - generate_json_from_template(xml_path, output_dir)
      - generate_xml_from_json(structure_path, params_table_path, stats_table_path, output_path)
      - update_json_from_scarab_params(scarab_param_path, mcpat_json_path)
      - update_json_from_scarab_stats(scarab_stats_path, mcpat_json_path)
    """

    def __init__(self):
        # Internal state
        self.structure = OrderedDict()
        self.params_table = OrderedDict()
        self.stats_table = OrderedDict()

    # === Static utilities ===
    @staticmethod
    def load_ordered_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f, object_pairs_hook=OrderedDict)

    @staticmethod
    def dump_ordered_json(obj, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    @staticmethod
    def print_xml(elem):
        rough = ET.tostring(elem, encoding="utf-8")
        parsed = minidom.parseString(rough)
        return parsed.toprettyxml(indent="  ")

    @staticmethod
    def get_clean_name(full_id):
        if full_id.startswith("system.core."):
            return full_id[len("system.core."):]
        elif full_id.startswith("system."):
            return full_id[len("system."):]
        return full_id

    # === Internal helpers ===
    def _parse_component_node(self, node):
        """
        Recursively parse <component> nodes in XML and populate param/stat tables.
        """
        comp_id = node.attrib["id"]
        comp_name = node.attrib.get("name", comp_id)
        children = OrderedDict()

        self.params_table[comp_id] = OrderedDict()
        self.stats_table[comp_id] = OrderedDict()

        for child in node:
            if child.tag == "param":
                name = child.attrib["name"]
                value = child.attrib["value"]
                self.params_table[comp_id][name] = value
                children[name] = ["param", None]
            elif child.tag == "stat":
                name = child.attrib["name"]
                value = child.attrib["value"]
                self.stats_table[comp_id][name] = value
                children[name] = ["stat", None]
            elif child.tag == "component":
                sub_id = child.attrib["id"]
                sub_node = self._parse_component_node(child)
                children[sub_id] = ["component", sub_node]

        return {
            "name": comp_name,
            "children": children
        }

    def _build_component_xml(self, comp_id, node_obj):
        """
        Recursively build <component> XML from internal JSON-like structure.
        """
        clean_name = self.get_clean_name(comp_id)
        comp_elem = ET.Element("component", id=comp_id, name=node_obj.get("name", clean_name))

        # Add param tags
        for key, val in self.params_table.get(comp_id, {}).items():
            ET.SubElement(comp_elem, "param", name=key, value=str(val))

        # Add stat tags
        for key, val in self.stats_table.get(comp_id, {}).items():
            ET.SubElement(comp_elem, "stat", name=key, value=str(val))

        # Add child components recursively
        for child_id, (tag_type, subnode) in node_obj.get("children", {}).items():
            if tag_type == "component":
                child_elem = self._build_component_xml(child_id, subnode)
                comp_elem.append(child_elem)

        return comp_elem

    def _map_scarab_params_to_mcpat_json(self, scarab_params, mcpat_json):
        """
        Internal helper:
        Given a dict of Scarab command-line parameters and an existing McPAT JSON,
        return a new McPAT JSON with values overwritten where mappings exist.
        """

        # Core-level parameters
        core = mcpat_json.setdefault("system.core0", {})

        core["issue_width"] = scarab_params.get("ISSUE_WIDTH", core.get("issue_width"))
        core["peak_issue_width"] = scarab_params.get("ISSUE_WIDTH", core.get("peak_issue_width"))
        core["decode_width"] = scarab_params.get("DECODE_WIDTH", core.get("decode_width"))
        core["commit_width"] = scarab_params.get("NODE_RET_WIDTH", core.get("commit_width"))

        core["ROB_size"] = scarab_params.get("NODE_TABLE_SIZE", core.get("ROB_size"))
        core["decoded_stream_buffer_size"] = scarab_params.get("IDQ_SIZE", core.get("decoded_stream_buffer_size"))
        rs_list = scarab_params.get("RS_SIZES", "").split()
        if len(rs_list) == 3:
            int_rs = int(rs_list[0])
            fp_rs = int(rs_list[1])
            mem_rs = int(rs_list[2])
            core["instruction_window_size"] = str(int_rs + mem_rs)
            core["fp_instruction_window_size"] = str(fp_rs)

        core["archi_Regs_IRF_size"] = "24"
        core["archi_Regs_FRF_size"] = "32"
        core["phy_Regs_IRF_size"] = scarab_params.get("REG_TABLE_INTEGER_PHYSICAL_SIZE", core.get("phy_Regs_IRF_size"))
        core["phy_Regs_FRF_size"] = scarab_params.get("REG_TABLE_VECTOR_PHYSICAL_SIZE", core.get("phy_Regs_FRF_size"))

        core["store_buffer_size"] = "114"
        core["load_buffer_size"] = "192"

        # I-Cache
        icache = mcpat_json.setdefault("system.core0.icache", {})
        icache_cfg = icache.get("icache_config", "").split(",")
        while len(icache_cfg) < 8:
            icache_cfg.append("1")
        if "ICACHE_SIZE" in scarab_params:
            icache_cfg[0] = str(scarab_params.get("ICACHE_SIZE"))
        if "ICACHE_LINE_SIZE" in scarab_params:
            icache_cfg[1] = str(scarab_params.get("ICACHE_LINE_SIZE"))
        if "ICACHE_ASSOC" in scarab_params:
            icache_cfg[2] = str(scarab_params.get("ICACHE_ASSOC"))
        icache["icache_config"] = ",".join(icache_cfg)

        # D-Cache
        dcache = mcpat_json.setdefault("system.core0.dcache", {})
        dcache_cfg = dcache.get("dcache_config", "").split(",")
        while len(dcache_cfg) < 8:
            dcache_cfg.append("1")
        if "DCACHE_SIZE" in scarab_params:
            dcache_cfg[0] = str(scarab_params.get("DCACHE_SIZE"))
        if "DCACHE_LINE_SIZE" in scarab_params:
            dcache_cfg[1] = str(scarab_params.get("DCACHE_LINE_SIZE"))
        if "DCACHE_ASSOC" in scarab_params:
            raw_assoc = int(scarab_params["DCACHE_ASSOC"])
            floor_power_of_two = lambda n: 1 << int(math.log2(max(n, 1)))
            dcache_cfg[2] = str(floor_power_of_two(raw_assoc))
        if "DCACHE_READ_PORTS" in scarab_params:
            dcache_cfg[3] = str(scarab_params.get("DCACHE_READ_PORTS"))
        if "DCACHE_WRITE_PORTS" in scarab_params:
            dcache_cfg[4] = str(scarab_params.get("DCACHE_WRITE_PORTS"))
        if "DCACHE_CYCLES" in scarab_params:
            dcache_cfg[5] = str(scarab_params.get("DCACHE_CYCLES"))
        dcache["dcache_config"] = ",".join(dcache_cfg)

        # Branch predictor (BTB entries)
        btb = mcpat_json.setdefault("system.core0.BTB", {})
        btb_cfg = btb.get("BTB_config", "").split(",")
        while len(btb_cfg) < 6:
            btb_cfg.append("1")
        btb_cfg[0] = str(scarab_params.get("BTB_ENTRIES"))
        btb["BTB_config"] = ",".join(btb_cfg)

        # L3 / MLC mapping
        l3 = mcpat_json.setdefault("system.L30", {})
        l3_cfg = l3.get("L3_config", "").split(",")
        while len(l3_cfg) < 8:
            l3_cfg.append("1")
        if "MLC_SIZE" in scarab_params:
            l3_cfg[0] = str(scarab_params.get("MLC_SIZE"))
        if "MLC_LINE_SIZE" in scarab_params:
            l3_cfg[1] = str(scarab_params.get("MLC_LINE_SIZE", 64))
        if "MLC_ASSOC" in scarab_params:
            l3_cfg[2] = str(scarab_params.get("MLC_ASSOC"))
        if "MLC_CYCLES" in scarab_params:
            l3_cfg[5] = str(scarab_params.get("MLC_CYCLES"))
        l3["L3_config"] = ",".join(l3_cfg)

        return mcpat_json

    def _map_scarab_stats_to_mcpat_json(self, scarab_stats, mcpat_json):
        """
        Internal helper:
        Given a dict of Scarab command-line power stats and an existing McPAT JSON,
        return a new McPAT JSON with values overwritten where mappings exist.
        """

        # --- System level ---
        system = mcpat_json.setdefault("system", {})
        system["total_cycles"] = str(scarab_stats.get("POWER_CYCLE_count", system.get("total_cycles")))
        system["busy_cycles"] = str(scarab_stats.get("POWER_CYCLE_count", system.get("busy_cycles")))

        # --- Core level ---
        core = mcpat_json.setdefault("system.core0", {})
        core["total_cycles"] = str(scarab_stats.get("POWER_CYCLE_count", system.get("total_cycles")))
        core["busy_cycles"] = str(scarab_stats.get("POWER_CYCLE_count", system.get("busy_cycles")))
        core["total_instructions"] = str(scarab_stats.get("POWER_OP_count", core.get("total_instructions")))
        core["uop_miss"] = str(scarab_stats.get("UOP_CACHE_MISS_count", core.get("uop_miss")))
        core["uop_hit"] = str(scarab_stats.get("UOP_CACHE_HIT_count", core.get("uop_hit")))
        core["uop_cache_access"] = str(scarab_stats.get("POWER_UOPCACHE_ACCESS_count", core.get("uop_cache_access")))
        core["uop_cache_miss"] = str(scarab_stats.get("POWER_UOPCACHE_MISS_count", core.get("uop_cache_miss")))
        core["int_instructions"] = str(scarab_stats.get("POWER_INT_OP_count", core.get("int_instructions")))
        core["fp_instructions"] = str(scarab_stats.get("POWER_FP_OP_count", core.get("fp_instructions")))
        core["branch_instructions"] = str(scarab_stats.get("POWER_BRANCH_OP_count", core.get("branch_instructions")))
        core["branch_mispredictions"] = str(scarab_stats.get("POWER_BRANCH_MISPREDICT_count", core.get("branch_mispredictions")))
        core["load_instructions"] = str(scarab_stats.get("POWER_LD_OP_count", core.get("load_instructions")))
        core["store_instructions"] = str(scarab_stats.get("POWER_ST_OP_count", core.get("store_instructions")))
        core["committed_instructions"] = str(scarab_stats.get("POWER_COMMITTED_OP_count", core.get("committed_instructions")))
        core["committed_int_instructions"] = str(scarab_stats.get("POWER_COMMITTED_INT_OP_count", core.get("committed_int_instructions")))
        core["committed_fp_instructions"] = str(scarab_stats.get("POWER_COMMITTED_FP_OP_count", core.get("committed_fp_instructions")))
        core["ROB_reads"] = str(scarab_stats.get("POWER_ROB_READ_count", core.get("ROB_reads")))
        core["ROB_writes"] = str(scarab_stats.get("POWER_ROB_WRITE_count", core.get("ROB_writes")))
        core["rename_reads"] = str(scarab_stats.get("POWER_RENAME_READ_count", core.get("rename_reads")))
        core["rename_writes"] = str(scarab_stats.get("POWER_RENAME_WRITE_count", core.get("rename_writes")))
        core["fp_rename_reads"] = str(scarab_stats.get("POWER_FP_RENAME_READ_count", core.get("fp_rename_reads")))
        core["fp_rename_writes"] = str(scarab_stats.get("POWER_FP_RENAME_WRITE_count", core.get("fp_rename_writes")))
        core["inst_window_reads"] = str(scarab_stats.get("POWER_INST_WINDOW_READ_count", core.get("inst_window_reads")))
        core["inst_window_writes"] = str(scarab_stats.get("POWER_INST_WINDOW_WRITE_count", core.get("inst_window_writes")))
        core["inst_window_wakeup_accesses"] = str(scarab_stats.get("POWER_INST_WINDOW_WAKEUP_ACCESS_count", core.get("inst_window_wakeup_accesses")))
        core["fp_inst_window_reads"] = str(scarab_stats.get("POWER_FP_INST_WINDOW_READ_count", core.get("fp_inst_window_reads")))
        core["fp_inst_window_writes"] = str(scarab_stats.get("POWER_FP_INST_WINDOW_WRITE_count", core.get("fp_inst_window_writes")))
        core["fp_inst_window_wakeup_accesses"] = str(scarab_stats.get("POWER_FP_INST_WINDOW_WAKEUP_ACCESS_count", core.get("fp_inst_window_wakeup_accesses")))
        core["int_regfile_reads"] = str(scarab_stats.get("POWER_INT_REGFILE_READ_count", core.get("int_regfile_reads")))
        core["float_regfile_reads"] = str(scarab_stats.get("POWER_FP_REGFILE_READ_count", core.get("float_regfile_reads")))
        core["int_regfile_writes"] = str(scarab_stats.get("POWER_INT_REGFILE_WRITE_count", core.get("int_regfile_writes")))
        core["float_regfile_writes"] = str(scarab_stats.get("POWER_FP_REGFILE_WRITE_count", core.get("float_regfile_writes")))
        core["function_calls"] = str(scarab_stats.get("POWER_FUNCTION_CALL_count", core.get("function_calls")))
        core["ialu_accesses"] = str(scarab_stats.get("POWER_IALU_ACCESS_count", core.get("ialu_accesses")))
        core["fpu_accesses"] = str(scarab_stats.get("POWER_FPU_ACCESS_count", core.get("fpu_accesses")))
        core["mul_accesses"] = str(scarab_stats.get("POWER_MUL_ACCESS_count", core.get("mul_accesses")))
        core["cdb_alu_accesses"] = str(scarab_stats.get("POWER_CDB_IALU_ACCESS_count", core.get("cdb_alu_accesses")))
        core["cdb_mul_accesses"] = str(scarab_stats.get("POWER_CDB_MUL_ACCESS_count", core.get("cdb_mul_accesses")))
        core["cdb_fpu_accesses"] = str(scarab_stats.get("POWER_CDB_FPU_ACCESS_count", core.get("cdb_fpu_accesses")))

        # --- ITLB ---
        itlb = mcpat_json.setdefault("system.core0.itlb", {})
        itlb["total_accesses"] = str(scarab_stats.get("POWER_ITLB_ACCESS_count", itlb.get("total_accesses")))

        # --- ICACHE ---
        icache = mcpat_json.setdefault("system.core0.icache", {})
        icache["read_accesses"] = str(scarab_stats.get("POWER_ICACHE_ACCESS_count", icache.get("read_accesses")))
        icache["read_misses"] = str(scarab_stats.get("POWER_ICACHE_MISS_count", icache.get("read_misses")))

        # --- DTLB ---
        dtlb = mcpat_json.setdefault("system.core0.dtlb", {})
        dtlb["total_accesses"] = str(scarab_stats.get("POWER_DTLB_ACCESS_count", dtlb.get("total_accesses")))

        # --- DCACHE ---
        dcache = mcpat_json.setdefault("system.core0.dcache", {})
        dcache["read_accesses"] = str(scarab_stats.get("POWER_DCACHE_READ_ACCESS_count", dcache.get("read_accesses")))
        dcache["write_accesses"] = str(scarab_stats.get("POWER_DCACHE_WRITE_ACCESS_count", dcache.get("write_accesses")))
        dcache["read_misses"] = str(scarab_stats.get("POWER_DCACHE_READ_MISS_count", dcache.get("read_misses")))
        dcache["write_misses"] = str(scarab_stats.get("POWER_DCACHE_WRITE_MISS_count", dcache.get("write_misses")))

        # --- BTB ---
        btb = mcpat_json.setdefault("system.core0.BTB", {})
        btb["read_accesses"] = str(scarab_stats.get("POWER_BTB_READ_count", btb.get("read_accesses")))
        btb["write_accesses"] = str(scarab_stats.get("POWER_BTB_WRITE_count", btb.get("write_accesses")))

        # --- L3 / LLC (system.L30) ---
        l3 = mcpat_json.setdefault("system.L30", {})
        l3["read_accesses"] = str(scarab_stats.get("POWER_LLC_READ_ACCESS_count", l3.get("read_accesses")))
        l3["write_accesses"] = str(scarab_stats.get("POWER_LLC_WRITE_ACCESS_count", l3.get("write_accesses")))
        l3["read_misses"] = str(scarab_stats.get("POWER_LLC_READ_MISS_count", l3.get("read_misses")))
        l3["write_misses"] = str(scarab_stats.get("POWER_LLC_WRITE_MISS_count", l3.get("write_misses")))

        # --- MC (memory controller) ---
        mc = mcpat_json.setdefault("system.mc", {})
        mc["memory_accesses"] = str(scarab_stats.get("POWER_MEMORY_CTRL_ACCESS_count", mc.get("memory_accesses")))
        mc["memory_reads"] = str(scarab_stats.get("POWER_MEMORY_CTRL_READ_count", mc.get("memory_reads")))
        mc["memory_writes"] = str(scarab_stats.get("POWER_MEMORY_CTRL_WRITE_count", mc.get("memory_writes")))

        return mcpat_json

    # === Public API ===
    def generate_json_from_template(self, xml_path, output_dir):
        """
        Given an XML template file, parse and export mcpat_structure.json,
        params_table.json, and stats_table.json into output_dir.
        """
        self.structure.clear()
        self.params_table.clear()
        self.stats_table.clear()

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for child in root:
            if child.tag == "component":
                cid = child.attrib["id"]
                self.structure[cid] = self._parse_component_node(child)

        os.makedirs(output_dir, exist_ok=True)
        self.dump_ordered_json(self.structure, os.path.join(output_dir, "mcpat_structure.json"))
        self.dump_ordered_json(self.params_table, os.path.join(output_dir, "params_table.json"))
        self.dump_ordered_json(self.stats_table, os.path.join(output_dir, "stats_table.json"))

        print(f"✅ JSON files exported to: {output_dir}")

    def generate_xml_from_json(self, structure_path, params_table_path, stats_table_path, output_path):
        """
        Given JSON files for structure, params_table, and stats_table, generate an XML file.
        """
        self.structure = self.load_ordered_json(structure_path)
        self.params_table = self.load_ordered_json(params_table_path)
        self.stats_table = self.load_ordered_json(stats_table_path)

        root = ET.Element("component", id="root", name="root")
        for comp_id, node_obj in self.structure.items():
            root.append(self._build_component_xml(comp_id, node_obj))

        xml_str = self.print_xml(root)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

        print(f"✅ XML generated and saved to: {output_path}")

    def update_json_from_scarab_params(self, scarab_params_path, mcpat_json_path):
        """
        Given a Scarab param text file and an McPAT JSON file,
        update the JSON in-place by applying Scarab values.
        This overwrites the original JSON file.
        """
        # Parse Scarab text file into dict
        scarab_params = OrderedDict()
        with open(scarab_params_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                if len(tokens) >= 2:
                    scarab_params[tokens[0]] = tokens[-1]

        # Apply mapping and overwrite the same file
        mcpat_data = self.load_ordered_json(mcpat_json_path)
        new_mcpat_data = self._map_scarab_params_to_mcpat_json(scarab_params, mcpat_data)
        self.dump_ordered_json(new_mcpat_data, mcpat_json_path)

        print(f"✅ McPAT Params JSON updated in-place: {mcpat_json_path}")

    def update_json_from_scarab_stats(self, scarab_stats_path, mcpat_json_path):
        """
        Given a pickled dict of power stats and an McPAT JSON file,
        update the JSON in-place by applying these averages.
        This overwrites the original JSON file.
        """
        # Load the stats from the .pkl
        import pickle
        with open(scarab_stats_path, "rb") as f:
            scarab_stats = pickle.load(f)

        # Apply mapping and overwrite the same file
        mcpat_data = self.load_ordered_json(mcpat_json_path)
        new_mcpat_data = self._map_scarab_stats_to_mcpat_json(scarab_stats, mcpat_data)
        self.dump_ordered_json(new_mcpat_data, mcpat_json_path)

        print(f"✅ McPAT Stats JSON updated in-place: {mcpat_json_path}")
