import os
import shutil
from datetime import datetime

# The purpose of this code was to find the log files created by LibSignal
# and extract metrics for each agent and scenario


def find_most_recent_log_file(logger_dir):
    """
    Find the most recent .log file in the specified logger directory.
    """
    try:
        log_files = [
            os.path.join(logger_dir, f) for f in os.listdir(logger_dir) if f.endswith(".log")
        ]
        if not log_files:
            return None
        most_recent_file = max(log_files, key=os.path.getmtime)
        return most_recent_file
    except Exception as e:
        print(f"Error accessing {logger_dir}: {e}")
        return None


def process_agent_folders(base_dir, output_dir):
    """
    Iteratively process folders to extract and copy the most recent log files.
    """
    for root, dirs, files in os.walk(base_dir):
        # Check for cityflow1x1 folders
        if "cityflow1x1" in root:
            logger_dir = os.path.join(root, "logger")
            if os.path.exists(logger_dir):
                # Find the most recent log file
                most_recent_file = find_most_recent_log_file(logger_dir)
                if most_recent_file:
                    # Determine the output file name
                    agent_folder = root.split(os.sep)[-3]  # Extract agent folder name
                    cityflow_folder = root.split(os.sep)[-2]  # Extract cityflow folder name
                    output_file_name = f"{agent_folder}_{cityflow_folder}.log"
                    output_file_path = os.path.join(output_dir, output_file_name)
                    # Copy the file
                    try:
                        shutil.copy(most_recent_file, output_file_path)
                        print(f"Copied {most_recent_file} to {output_file_path}")
                    except Exception as e:
                        print(f"Error copying file {most_recent_file}: {e}")


if __name__ == "__main__":
    base_directory = "/home/jouchuki/DaRL/LibSignal/data/output_data/tsc"  # Replace with the base directory path
    output_directory = "./extracted_logs"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process agent folders
    process_agent_folders(base_directory, output_directory)
