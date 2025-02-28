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
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Read descriptor file name')
    parser.add_argument('-d','--descriptor_name', required=True, help='Experiment descriptor name. Usage: -d exp2.json')
    parser.add_argument('-a','--application_name', required=True, help='Application name. Usage: -a simple_multi_update')

    # Parse the command-line arguments
    args = parser.parse_args()

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
            exp_path = str(os.getenv('HOME')) + '/simpoint_flow/simulations/' + workload + '/' + experiment + '/' + config_key
            print(exp_path)
            if not os.path.exists(exp_path):
              print(f"The experiment does not exist!")
              return None
            config_value = descriptor_data["configurations"][config_key]
            if args.application_name == "allbench":
                command = 'python3 /usr/local/bin/gather_cluster_results.py /simpoint_traces/' + workload + '/simpoints/ $HOME/simpoint_flow/simulations/' + workload + '/' + experiment + '/' + config_key
            else:
                command = 'python3 /usr/local/bin/gather_cluster_results.py $HOME/simpoint_flow/' + workload + '/simpoints/ $HOME/simpoint_flow/simulations/' + workload + '/' + experiment + '/' + config_key
            os.system(command)

if __name__ == "__main__":
    run_experiment()
