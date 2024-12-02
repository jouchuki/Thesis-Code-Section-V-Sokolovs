# Thesis-Code-Section-V-Sokolovs
Code Section of the Thesis, student name Vladislavs Sokolovs, snr 2066744

## Attribution
The `SBPPO` agent class implementation in this project was adapted from the LibSignal framework. 
Documentation and examples from LibSignal were used as references to create a 
PPO-compatible agent that integrates with LibSignal's Registry system and 
environment setup. 

Significant adaptations include:
- Integration of Registry for parameter mapping.
- Custom Gym environment `SingleAgentEnv` for use with Stable Baselines3 (SB3).
- Adapting agent methods for use with Stable Baselines3 (SB3).

LibSignal Documentation: [[https://darl-libsignal.github.io/LibSignalDoc/content/getstart/Start.html](https://darl-libsignal.github.io/LibSignalDoc/content/getstart/Introduction.html)]


## Libraries

All libraries used in the project with their respective versions are available in the thesis_env.yml file (except LibSignal, because it was cloned with git and not built in Python).

To recreate environment, use:

```
conda env create -f thesis_env.yml
```

Then, clone LibSignal repository

```
git clone git@github.com:DaRL-LibSignal/LibSignal.git
```

## Instructions

To recreate the results of this thesis, you have to do the following:
Please, copy sb_ppo.yml file to ```./LibSignal/configs/tsc``` and the ```sb_ppo.py``` file to ```./LibSignal/agent```.
Import SBPPO class in the ```./LibSignal/agent/__init__.py ```


First, make sure that for training the configuration files contain right flags.
This means, that in ```dqn.yml``` and ```sb_ppo.yml``` in the ```trainer``` section the parameter train_model should be True, and load_model should be False.
Then, to train the models, run the following command in the LibSignal directory:
```
python run.py -a dqn -n cityflow1x1
python run.py -a sb_ppo -n cityflow1x1
```
To test the models, change train_model to False and load_model to True.
Put ```actual_run.py file``` in the LibSignal directory and run it.
This will create evaluations for each agent and scenario described in the thesis.

The logs will be created for each simulation via LibSignal.
Run ```get_all_logs.py``` to extract the most recent logs for every agent.

Then run ```logs_parser.py``` to create visualisations of training performance.

Run ```calculate_arpl.py``` to calculate arrival rates for each scenarion explored in the thesis

Run ```to_table.py``` to create a LaTeX table of performance metric for Config2, Config3, Config4 for each explored agent.


## References:
Mei, H., Lei, X., Da, L., Shi, B., & Wei, H. (2024). Libsignal: An open library for traffic signal control. Machine Learning, 113(8), 5235-5271. Retrieved from [https://github.com/DaRL-LibSignal/LibSignal]
Zhang, H., Feng, S., Liu, C., Ding, Y., Zhu, Y., Zhou, Z., ... & Li, Z. (2019, May). Cityflow: A multi-agent reinforcement learning environment for large scale city traffic scenario. In The world wide web conference (pp. 3620-3624). Version 0.1. Retrieved from [https://github.com/cityflow-project/CityFlow]
Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268), 1–8. Retrieved
from http://jmlr.org/papers/v22/20-1364.html. Version (2.3.2). Installed with pip.

