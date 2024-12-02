import os
import re
from collections import defaultdict

# This script creates a performance metrics table (Table 2).
# It first extracts metrics from .log files
# Organizes it
# and produces a .tex file of the resulting table


def extract_metrics_from_log(log_file_path):
    """
    Extract metrics from a log file.
    """
    metrics = {}
    try:
        with open(log_file_path, 'r') as file:
            content = file.read()
            # Regular expressions to extract metrics
            travel_time_match = re.search(r'Final Travel Time is ([\d\.]+)', content)
            queue_match = re.search(r'queue: ([\d\.]+)', content)
            delay_match = re.search(r'delay: ([\d\.]+)', content)
            throughput_match = re.search(r'throughput: (\d+)', content)

            if travel_time_match:
                metrics['Average Travel Time'] = float(travel_time_match.group(1))
            if queue_match:
                metrics['Queue'] = float(queue_match.group(1))
            if delay_match:
                metrics['Delay'] = float(delay_match.group(1))
            if throughput_match:
                metrics['Throughput'] = int(throughput_match.group(1))
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
    return metrics


def collect_log_files(logs_directory):
    """
    Collect all log files from the directory.
    """
    log_files = []
    for filename in os.listdir(logs_directory):
        if filename.endswith('.log'):
            log_files.append(os.path.join(logs_directory, filename))
    return log_files


def parse_filename(filename):
    """
    Parse the agent and config from the filename.
    """
    # Example filename: cityflow_sb_ppo_cityflow1x1_config2.log
    basename = os.path.basename(filename)
    match = re.match(r'cityflow_(.+?)_cityflow1x1_(config\d)\.log', basename)
    if match:
        agent = match.group(1)
        config = match.group(2)
        return agent, config
    else:
        return None, None


def organize_data(log_files):
    """
    Organize the extracted metrics into a structured format.
    """
    data = defaultdict(lambda: defaultdict(dict))
    for log_file in log_files:
        agent, config = parse_filename(log_file)
        if agent and config:
            metrics = extract_metrics_from_log(log_file)
            if metrics:
                data[config][agent] = metrics
    return data


def generate_latex_table(data):
    """
    Generate LaTeX code for the table using APA style.
    """
    agents = ['sb_ppo', 'dqn', 'sotl', 'maxpressure', 'fixedtime']
    metrics_names = ['Average Travel Time', 'Queue', 'Delay', 'Throughput']
    configs = ['config1', 'config2', 'config3', 'config4']

    latex_code = "\\begin{table}[h!]\n\\centering\n"
    latex_code += "\\caption{Performance Metrics for Different Agents and Configurations}\n"
    latex_code += "\\label{tab:performance_metrics}\n"
    latex_code += "\\begin{tabular}{lcccc}\n"
    latex_code += "\\toprule\n"
    latex_code += "Agent & Average Travel Time & Queue & Delay & Throughput \\\\\n"
    latex_code += "\\midrule\n"

    for config in configs:
        latex_code += f"\\multicolumn{{5}}{{l}}{{\\textbf{{Configuration: {config.capitalize()}}}}} \\\\\n"
        latex_code += "\\midrule\n"
        for agent in agents:
            metrics = data.get(config, {}).get(agent, {})
            row = f"{agent.replace('sb_', '').capitalize()}"
            for metric in metrics_names:
                value = metrics.get(metric, 'N/A')
                row += f" & {value}"
            row += " \\\\\n"
            latex_code += row
        latex_code += "\\midrule\n"

    latex_code += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return latex_code


def main():
    logs_directory = './raw_logs/extracted_logs/'
    log_files = collect_log_files(logs_directory)
    data = organize_data(log_files)
    latex_table_code = generate_latex_table(data)
    # Write LaTeX code to a file
    with open('performance_metrics_table.tex', 'w') as f:
        f.write(latex_table_code)
    print("LaTeX table code has been written to 'performance_metrics_table.tex'.")


if __name__ == '__main__':
    main()
