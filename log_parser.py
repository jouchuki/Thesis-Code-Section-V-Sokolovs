import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


matplotlib.use('agg')

# The purpose of this file was to create visualisations of the performance during training
# The resulting figures were included in the thesis in the Fig. 1

def list_log_files(directory):
    try:
        # List all .log files in the directory
        log_files = [directory+"/"+file for file in os.listdir(directory) if file.endswith('.log')]
        return log_files
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def parse_to_dict(fn):
    # Parse the metrics from logs and extract only the relevant ones
    fdict = {}
    print(fn)
    for file in fn:
        with open(file, 'r') as f:
            flist = []
            for ln in f.readlines():
                x = ln.strip().split('\t')
                flist.append(x)
                # print(x)
                # 0 - agentname
                # 1 - TRAIN or TEST
                # 2 - ep number
                # 3 - av travel time
                # 4 - rewards
                # 5 - queue
                # 6 - delay
                # 7 - throughput
        trainlist = [flist[i] for i in range(0, len(flist)-1, 2)]
        testlist = [flist[i] for i in range(1, len(flist), 2)]

        assert all(testlist[i][1] == 'TEST' for i in range(len(testlist)))
        print("Assertion successful")
        assert all(trainlist[i][1] == 'TRAIN' for i in range(len(testlist)))
        print("Assertion successful")

        proxy = [item[3:4] + item[6:] for item in testlist]

        fdict.update({testlist[0][0]: proxy})

        # 0 - av travel time
        # 1 - queue
        # 2 - delay
        # 3 - throughput
        print(fdict)
    return fdict


def restructure_data_per_metric(data):
    restructured_data = {}
    for agent in data:
        num_metrics = len(data[agent][0])
        metrics_over_episodes = [[] for _ in range(num_metrics)]
        for episode in data[agent]:
            for metric_index, metric_value in enumerate(episode):
                metrics_over_episodes[metric_index].append(float(metric_value))
        restructured_data[agent] = metrics_over_episodes
    return restructured_data


def plot_metrics(data):
    agents = list(data.keys())
    num_metrics = len(data[agents[0]])  # Number of metrics
    num_episodes = len(data[agents[0]][0])  # Number of episodes

    metrics = ['Average Travel Time', 'Queue', 'Delay', 'Throughput']

    for metric_index, metric in zip(range(num_metrics), metrics):
        plt.figure(figsize=(8, 6))
        for agent in agents:
            metric_values = data[agent][metric_index]
            plt.plot(range(1, num_episodes + 1), metric_values, label=f'{agent}')
        plt.title(f'{metric} Over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel(f'{metric}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{metric}_performance.png')
        plt.close()


if __name__ == "__main__":
    logs = list_log_files("C:/Users/vsoko/OneDrive/Desktop/results_thesis_section/raw_logs")
    d = parse_to_dict(logs)
    d = restructure_data_per_metric(d)
    # print(d['dqn'][0])
    plot_metrics(d)