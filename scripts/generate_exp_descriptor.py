import argparse
import json

def generate_descriptor(args):
    # Create a dictionary with keys and values from command-line arguments
    descriptor_data = {"architecture": args.architecture, "workloads_list": args.workloads_list, "experiment": args.experiment}

    # Create a dictionary of configurations
    configuration_data = {"baseline": args.base_params}
    for value in args.sweep_values:
        key = args.sweep_param + "." + str(value)
        configuration_data[key] = args.base_params + " --" + args.sweep_param + " " + value

    descriptor_data["configurations"] = configuration_data

    return descriptor_data

def save_descriptor_to_json(descriptor_data, filename="experiment.json"):
    # Save the descriptor data to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(descriptor_data, json_file, indent=4)
    print(f"Descriptor saved to {filename}")

def main():
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Generate a JSON descriptor file.')
    parser.add_argument('-a','--architecture', required=True, help='Default CPU Architecture (kaby_lake, sunny_cove) Usage: -a sunny_cove')
    parser.add_argument('-w','--workloads_list', nargs='+', required=True, help='Workload list to test (clang,gcc,memcached,mongodb,mysql,postgres,redis,rocksdb,verilator,xgboost) Usage: -w mongodb mysql verilator xgboost')
    parser.add_argument('-e','--experiment', required=True, help='Experiment name. Usage: -e exp2')
    parser.add_argument('-b','--base_params', required=True, help='Baseline Scarab parameters on top of the given CPU architecture. Usage: -b "--fdip_enable 1 --icache_size 65536"')
    # Allow the user to provide additional key-value pairs
    parser.add_argument('-s','--sweep_param', required=False, help='Scarab parameter name to sweep. Usage: -s fe_ftq_block_num')
    parser.add_argument('-v','--sweep_values', nargs='+', required=False, help='A list of sweeping values of sweep_param. Usage: -v 2 4 6 8 10 12')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Generate descriptor data
    descriptor_data = generate_descriptor(args)

    # Save descriptor data to a JSON file
    save_descriptor_to_json(descriptor_data, args.experiment + ".json")

if __name__ == "__main__":
    main()
