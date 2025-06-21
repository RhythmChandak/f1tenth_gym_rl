import gymnasium as gym
import numpy as np
from .observation import Observation
from .rewards import Reward
from .f110_env import F110Env
from .base_classes import Simulator
from . import normalization
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

    def get_reward(self, obs, action):
        """
        Compute the reward based on the observation and action.
        
        :param obs: The observation from the environment.
        :param action: The action taken by the agent.
        :return: The computed reward.
        """
        # Calculate the deviation penalty
        deviation_penalty = self.calc_deviation_penalty(obs['deviation'])

        # Calculate the relative heading penalty
        rel_heading_penalty = self.calc_rel_heading_penalty(obs['rel_heading'])

        # Calculate the velocity reward
        velocity_reward = self.get_reward_speed(obs['longitudinal_vel'])

        # Calculate the collision penalty
        reward_collision = self.get_reward_collision(obs['collision'])

        # Calculate the advancement reward
        reward_advancement = self.get_reward_advancement(obs['advancement'])

        # Combine all rewards
        reward_positive = reward_advancement + velocity_reward
        reward_negative = deviation_penalty + rel_heading_penalty 
        reward = reward_positive + reward_negative(reward_positive) + reward_collision

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
        self.reward_type = TrajBasedReward(env=self, config_args=self.config_args)  


    def step(self, action):
        """
        Execute one time step within the environment based on the provided action.
        
        :param action: The action to be taken by the agent.
        """
        
        if action.shape == (2,):
            action = action.reshape((1, 2))
        #print(action.shape)

        norm_action = action.copy()
        action = self.denorm_action(action)

        # call parent's step
        obs, _, done, truncated, info = super(TcDriverRLEnv, self).step(action=action)

        # TODO: Call custom reward
        reward = self.reward_type.get_reward(obs, norm_action)

        obs_vec = self.vectorize_obs(obs,norm_action)

        return obs_vec, reward, done, truncated, info
    
    
    
    



