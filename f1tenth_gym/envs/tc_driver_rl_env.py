import gymnasium as gym
import numpy as np
import random
from .observation import Observation
from .rewards import Reward
from .f110_env import F110Env
from .base_classes import Simulator
from .normalization import denorm_action, normalise_observation, normalise_action
from .observation import observation_factory


class TrajBasedReward(Reward):
    """
    Trajectory-based reward class for the F110 Gym environment.
    This class computes rewards based on the agent's trajectory and its deviation from a reference trajectory.
    """

    def __init__(self, env, config_args={}):
        super().__init__(env, config_args)
        self.env = env
        self.config_args = config_args
        

    def get_reward(self, obs, action, prev_obs=None, prev_action=None):
        """
        Compute the reward based on the observation and action.
        
        :param obs: The observation from the environment.
        :param action: The action taken by the agent.
        :return: The computed reward.
        """
        # Calculate the deviation penalty
        deviation_penalty = self.calc_deviation_penalty(obs['deviation'])

        # Calculate the relative heading penalty
        # rel_heading_penalty = self.calc_rel_heading_penalty(obs['rel_heading'])

        # Action smoothing penalty (optional)
        action_smoothing_penalty = self.calc_action_smoothness_penalty(action, prev_action) if prev_action is not None else 0.0

        # Calculate the velocity reward
        velocity_reward = self.get_reward_speed(obs['longitudinal_vel'])

        # Calculate the collision penalty
        reward_collision = self.get_reward_collision(obs['collision'])

        # Calculate the advancement reward
        reward_advancement = self.get_reward_adv(obs['progress_along_track'], prev_obs['progress_along_track'] if prev_obs else 0)

        # Combine all rewards
        reward_positive = reward_advancement + velocity_reward
        reward_negative = deviation_penalty + action_smoothing_penalty
        reward = reward_positive + reward_negative + reward_collision

        
        return reward

        
class TcDriverRLEnv(F110Env):
    """
    F110 Gym environment for RL agents using the TC Driver simulator.
    This environment is designed to be compatible with the TC Driver simulator and provides a structured interface for RL training.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_type = observation_factory(env=self, type='traj_based')
        self.observation_space = self.observation_type.space()
        self.reward_type = TrajBasedReward(env=self, config_args=self.config)  
        self.last_action = np.array([[0.0, 0.0]])
        self.last_observation = None

    def step(self, action):
        """
        Execute one time step within the environment based on the provided action.
        
        :param action: The action to be taken by the agent.
        """
        
        if action.shape == (2,):
            action = action.reshape((1, 2))
        #print(action.shape)

        norm_action = normalise_action(action.copy(), self.config["params"])
        # action = denorm_action(action, self.config["params"])
        
        # call parent's step
        obs, _, done, truncated, info = super(TcDriverRLEnv, self).step(action=action)

        # TODO: Call custom reward
        if (len(obs.keys()) ==1):
            obs = obs[self.agent_ids[0]]
        reward = self.reward_type.get_reward(obs, action, self.last_observation, self.last_action)

        self.last_action = action
        self.last_observation = obs

        obs_vec = self.observation_type.vectorize_obs(obs,norm_action)

        return obs_vec, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and return the initial observation.
        
        :param seed: Optional seed for random number generation.
        :param options: Optional dictionary of additional options.
        :return: The initial observation after reset.
        """
        print("Resetting environment")
        self.last_action = np.array([[0.0, 0.0]])
        self.last_observation = None

        random_s = random.uniform(0, self.env.track.raceline.ss[-1])
        random_d = random.uniform(-self.config["params"]["width"], self.config["params"]["width"])
        random_phi = random.uniform(-np.pi/4, np.pi/4)

        random_pose = np.array(self.track.frenet_to_cartesian(random_s, random_d, random_phi)).reshape(1, 3)
        if options is None:
            options = {}
        options["poses"] = random_pose

        obs, info = super().reset(seed=seed, options=options)
        return obs, info
    