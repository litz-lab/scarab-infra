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
# File: client.py
# Author: Yinyuan Zhao (Litz Lab)
#
# ----------------------------------------------------------------------------

import argparse
from converter import Converter


def main():
    parser = argparse.ArgumentParser(description="Client for McPAT–Scarab format conversion workflow")
    parser.add_argument(
        "command",
        choices=["generate-json", "update-params", "update-stats", "generate-xml", "all"],
        help="Step to execute"
    )
    args = parser.parse_args()

    converter = Converter()

    if args.command == "generate-json":
        converter.generate_json_from_template(
            xml_path="./xml/template.xml",
            output_dir="./json"
        )
        print("✅ Generated JSON files from McPAT template.")

    elif args.command == "update-params":
        converter.update_json_from_scarab_params(
            scarab_params_path="./scarab_output/PARAMS.out",
            mcpat_json_path="./json/params_table.json"
        )
        print("✅ Updated JSON with Scarab PARAMS file.")

    elif args.command == "update-stats":
        converter.update_json_from_scarab_stats(
            scarab_stats_path="./scarab_output/power.pkl",
            mcpat_json_path="./json/stats_table.json"
        )
        print("✅ Updated JSON with Scarab STATS file.")

    elif args.command == "generate-xml":
        converter.generate_xml_from_json(
            structure_path="./json/mcpat_structure.json",
            params_table_path="./json/params_table.json",
            stats_table_path="./json/stats_table.json",
            output_path="./xml/mcpat_infile.xml"
        )
        print("✅ Generated McPAT XML input file.")

    elif args.command == "all":
        print("🚀 Running full McPAT–Scarab conversion pipeline...")

        converter.generate_json_from_template(
            xml_path="./xml/template.xml",
            output_dir="./json"
        )
        print("✅ Step 1: Generated JSON files from McPAT template.")

        converter.update_json_from_scarab_params(
            scarab_params_path="./scarab_output/PARAMS.out",
            mcpat_json_path="./json/params_table.json"
        )
        print("✅ Step 2: Updated JSON with Scarab PARAMS file.")

        converter.update_json_from_scarab_stats(
            scarab_stats_path="./scarab_output/power.pkl",
            mcpat_json_path="./json/stats_table.json"
        )
        print("✅ Step 3: Updated JSON with Scarab STATS file.")

        converter.generate_xml_from_json(
            structure_path="./json/mcpat_structure.json",
            params_table_path="./json/params_table.json",
            stats_table_path="./json/stats_table.json",
            output_path="./xml/mcpat_infile.xml"
        )
        print("✅ Step 4: Generated final McPAT XML input file.")
        print("🎉 All steps completed successfully.")


if __name__ == "__main__":
    main()
