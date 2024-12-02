import os
import json
import math
from functools import reduce
# The point of this code is to calculate arrival rates of the roads of every configuration.


def calculate_arrival_rates(flow_file):
    with open(flow_file, 'r') as f:
        flows = json.load(f)

    road_vehicle_counts = {}
    max_end_time = 0

    for flow in flows:
        route = flow['route']
        start_time = flow['startTime']
        end_time = flow['endTime']
        interval = flow['interval']

        starting_road = route[0]

        # Update the maximum end time
        if end_time > max_end_time:
            max_end_time = end_time

        # Calculate the number of vehicles generated by this flow
        if start_time == end_time:
            num_vehicles = 1
        else:
            num_vehicles = math.floor((end_time - start_time) / interval) + 1

        # Accumulate the number of vehicles for the starting road
        if starting_road in road_vehicle_counts:
            road_vehicle_counts[starting_road] += num_vehicles
        else:
            road_vehicle_counts[starting_road] = num_vehicles

    # Avoid division by zero if the simulation time is zero
    if max_end_time == 0:
        max_end_time = 1  # Assuming at least 1 second of simulation time

    # Calculate arrival rates per road
    arrival_rates = {}
    for road, vehicle_count in road_vehicle_counts.items():
        arrival_rate = (vehicle_count / max_end_time) * 3600  # Convert to vehicles per hour
        arrival_rates[road] = arrival_rate

    return arrival_rates


if __name__ == "__main__":
    flow_file = str(input("Path to flow.json: "))  # Input for the path to flow.json
    arrival_rates = calculate_arrival_rates(flow_file)

    # Calculate mean arrival rate
    mean = float(reduce(lambda x, y: x + y, list(arrival_rates.values()))) / len(arrival_rates.keys())

    # Determine parent directory of flow.json
    parent_dir = os.path.dirname(flow_file)
    output_file = os.path.join(parent_dir, "arrival_metrics.txt")

    # Write metrics to the output file
    with open(output_file, 'w') as f:
        for road, rate in arrival_rates.items():
            f.write(f"Road {road}: Arrival Rate = {rate:.2f} vehicles/hour\n")
        f.write(f"Intersection mean arrival rate = {mean:.2f} vehicles/hour\n")

    print(f"Metrics have been written to {output_file}")
