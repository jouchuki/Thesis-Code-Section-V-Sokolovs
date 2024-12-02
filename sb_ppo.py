"""
This implementation of the SBPPO class was adapted from the LibSignal framework.
Significant modifications were made to ensure compatibility with the
specific requirements of this project, including integration with LibSignal's
Registry mappings and tailored environment setup for PPO.

References:
LibSignal Documentation: https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
"""



from . import RLAgent
from common.registry import Registry
import numpy as np

from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator

from stable_baselines3 import PPO
import torch.nn as nn

import gym
import os

# Adapted from the LibSignal framework to define the SBPPO agent.
# Modifications include the use of Registry mappings for dynamic settings
# and the custom environment 'SingleAgentEnv' for Gym compatibility.

# The model has to be registered in the registry to be recognised by the library,
# so that the config file could be found
#                            |
#                            v
@Registry.register_model('sb_ppo')
class SBPPO(RLAgent):
    def __init__(self, world, rank):
        """
        SBPPO Agent

        This class implements a Proximal Policy Optimization (PPO) agent compatible
        with the LibSignal framework and Stable Baselines3 (SB3). It integrates with
        a custom Registry for dynamic configuration and features tailored Gym-compatible
        environments and generators.

        Attributes:
            world (object): The simulation world object.
            rank (int): The rank of the intersection being controlled.
            buffer_size (int): The size of the replay buffer.
            phase (bool): Whether to include phase information in observations.
            one_hot (bool): Whether to use one-hot encoding for phases.
            ob_length (int): Length of the observation vector.
        """
        # Adapted for SB3 using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        super().__init__(world, world.intersection_ids[rank])
        self.world = world
        self.rank = rank

        # Retrieve settings from Registry
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['setting'].param
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']

        # Initialize generators
        inter_id = self.world.intersection_ids[self.rank]
        self.inter = self.world.id2intersection[inter_id]
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.inter.phases))

        # Define observation space length
        if self.phase:
            if self.one_hot:
                self.ob_length = self.ob_generator.ob_length + len(self.inter.phases)
            else:
                self.ob_length = self.ob_generator.ob_length + 1
        else:
            self.ob_length = self.ob_generator.ob_length

        # Build the environment and model
        self.env = self._build_env()
        self.model = self._build_model()

    def _build_env(self):
        """
        Build the Gym-compatible environment for the agent.

        This method defines a custom `SingleAgentEnv` class that implements
        the Gym environment interface.

        Returns:
            gym.Env: A Gym-compatible environment instance.
        """

        # Adapted for SB3 using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html

        # Create a Gym environment
        class SingleAgentEnv(gym.Env):
            def __init__(env_self):
                super(SingleAgentEnv, env_self).__init__()
                env_self.action_space = self.action_space
                env_self.observation_space = gym.spaces.Box(
                    low=0, high=1, shape=(self.ob_length,), dtype=np.float32
                )

            def reset(env_self):
                # Reset the world state
                self.world.reset()
                obs = self.get_ob()
                return obs

            def step(env_self, action):
                # Apply the action to the world
                self.world.step([action])
                obs = self.get_ob()
                reward = self.get_reward()
                done = False
                info = {}
                return obs, reward, done, info

        # Instantiate the environment
        env = SingleAgentEnv()

        return env

    def _build_model(self):
        """
        Build the PPO model using Stable Baselines3.

        This method initializes a PPO model with custom policy settings.

        Returns:
            PPO: The PPO model instance.
        """
        # Adapted for SB3 using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        # Define policy kwargs
        policy_kwargs = dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            activation_fn=nn.ReLU,
        )

        # Initialize the PPO model
        model = PPO(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=self.learning_rate,
            verbose=1,
            policy_kwargs=policy_kwargs
        )
        return model

    def get_ob(self):
        # Get observation of the world.
        # Implemented using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_phase(self):
        # Get phase of the intersection.
        # Implemented using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        phase = []
        phase.append(self.phase_generator.generate())
        phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_reward(self):
        # Get reward from the reward generators.
        # Implemented using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        rewards = []
        rewards.append(self.reward_generator.generate())
        norm_rewards = [np.clip(r/224, -4, 4) for r in rewards]
        rewards = np.squeeze(np.array(norm_rewards))
        return rewards

    def get_queue(self):
        # get queue length metric.
        # Implemented using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        queue = []
        queue.append(self.queue.generate())
        queue = np.sum(np.squeeze(np.array(queue)))
        return queue

    def get_delay(self):
        # get delay metric
        # Implemented using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        delay = []
        delay.append(self.delay.generate())
        delay = np.sum(np.squeeze(np.array(delay)))
        return delay  

    def train(self, total_timesteps=3600):
        # Train the model for one iteration
        # Adapted for SB3 using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        self.model.learn(total_timesteps=total_timesteps)

    def get_action(self, obs):
        # Get an action from the model.
        # Adapted for SB3 using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        action, _states = self.model.predict(obs, deterministic=True)
        return action

    def save_model(self, e):
        # Save model.
        # Adapted for SB3 using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}')
        self.model.save(model_name)

    def load_model(self, e):
        # Load model.
        # Adapted for SB3 using examples from:
        # https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Agent.html
        model_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                  'model', f'{e}_{self.rank}')
        self.model = PPO.load(model_name, env=self.env)

    def remember(self, *args, **kwargs):
        pass

    def update_target_network(self):
        pass

    def do_observe(self, *args, **kwargs):
        pass