import json
import argparse
import os

def read_descriptor_from_json(filename="experiment.json"):
    # Read the descriptor data from a JSON file
    try:
        with open(filename, 'r') as json_file:
            descriptor_data = json.load(json_file)
        return descriptor_data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{filename}': {e}")
        return None

def run_experiment():
    # run_exp_using_descriptor.py
    # -d $EXPERIMENT.json
    # -a $APPNAME -g $APP_GROUPNAME : pt_traces
    # -c $BINCMD
    # -m $SCARABMODE &
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Read descriptor file name')
    parser.add_argument('-d','--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp2.json')
    parser.add_argument('-a','--application_name', required=True, help='Application name. Usage: -a simple_multi_update')
    parser.add_argument('-g','--application_group_name', required=True, help='Application group name. Usage: -g mongodb')
    parser.add_argument('-c','--binary_command', required=False, help='Binary command. Usage -c /usr/bin/mongd --config /etc/mongod.conf')
    parser.add_argument('-m','--scarab_mode', required=True, help='Scarab mode. Usage -m 2')

    # Parse the command-line arguments
    args = parser.parse_args()

    assert args.scarab_mode in ["0", "1", "2", "3", "4", "5"]

    # Specify the filename of the JSON descriptor file
    descriptor_filename = args.descriptor_name

    # Read descriptor data from the JSON file
    descriptor_data = read_descriptor_from_json(descriptor_filename)

    # Check if reading was successful
    if descriptor_data is not None:
        print("Descriptor data read successfully:")
        print(descriptor_data)

    architecture = descriptor_data["architecture"]
    experiment = descriptor_data["experiment"]

    # Run Scarab
    for workload in descriptor_data["workloads_list"]:
        for config_key in descriptor_data["configurations"].keys():
            if args.scarab_mode == '4':
              exp_path = str(os.getenv('HOME')) + '/simpoint_flow/simulations/' + workload + '/' + experiment + '/' + config_key
              print(exp_path)
              if os.path.exists(exp_path+'/ipc.csv'):
                print(f"The experiment already exists! Change the experiment name.")
                continue
            if args.scarab_mode == '5':
              exp_path = str(os.getenv('HOME')) + '/nonsimpoint_flow/simulations/' + workload + '/' + experiment + '/' +config_key
              print(exp_path)
              if os.path.exists(exp_path+'/memory.stat.0.csv'):
                print(f"The experiment already exists! Change the experiment name.")
                continue
            config_value = descriptor_data["configurations"][config_key]
            if args.application_name == "allbench" or args.application_name == "isca2024":
                if workload in ["602.gcc_s", "clang", "gcc", "mongodb", "mysql", "postgres", "verilator", "xgboost"]:
                    use_traces_simp = "1"
                elif workload in ["500.perlbench_r", "502.gcc_r", "505.mcf_r", "520.omnetpp_r", "523.xalancbmk_r", "525.x264_r", "531.deepsjeng_r", "541.leela_r", "548.exchange2_r", "557.xz_r",
                                "503.bwaves_r", "507.cactuBSSN_r", "508.namd_r", "510.parest_r", "511.povray_r", "519.lbm_r", "521.wrf_r", "526.blender_r", "527.cam4_r", "538.imagick_r", "544.nab_r", "549.fotonik3d_r", "554.roms_r"]:
                    use_traces_simp = "2"
                else:
                    use_traces_simp = "0"
                command = 'run_scarab_allbench.sh "' + workload + '" "' + args.application_group_name + '" "" "' + experiment + '/' + config_key + '" "' + config_value + '" "' + args.scarab_mode + '" "' + architecture + '" "' + use_traces_simp + '"'
            else:
                command = 'run_scarab.sh "' + args.application_name + '" "' + args.application_group_name + '" "' + args.binary_command + '" "' + experiment + '/' + config_key + '" "' + config_value + '" "' + args.scarab_mode + '" "' + architecture + '"'
            os.system(command)

if __name__ == "__main__":
    run_experiment()
