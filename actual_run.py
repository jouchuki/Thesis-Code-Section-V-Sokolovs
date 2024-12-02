import subprocess

# Define the argument combinations
# These two lists define agents of interest and networks the agents will be tested on
agents = ["sb_ppo", "dqn", "fixedtime", "maxpressure", "sotl"]
networks = [
    "cityflow1x1",
    "cityflow1x1_config2",
    "cityflow1x1_config3",
    "cityflow1x1_config4",
]

# Base command
base_command = ["python", "run.py"]

# Iterate over all combinations of -a and -n arguments
for agent in agents:
    for network in networks:
        command = base_command + ["-a", agent, "-n", network]
        print(f"Running: {' '.join(command)}")
        try:
            # Execute the command
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {' '.join(command)}")
            print(f"Error message: {e}")
