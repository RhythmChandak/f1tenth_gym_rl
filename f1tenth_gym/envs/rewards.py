import numpy as np
from abc import abstractmethod
import logging

class Reward:
    """
    Base class for rewards in the F1Tenth Gym environment for RL agents.
    This class provides a structure for defining rewards based on the state of the environment.
    """
    def __init__(
            self,
            env,
            config_args={},
    ):
        """
        Initialize the Reward class.

        :param env: The environment instance.
        :param config_args: Configuration arguments for the reward function.
        """
        self.env = env
        self.config_args = config_args

    # abstract method for get reward
    @abstractmethod
    def get_reward(self, obs, action):
        """
        Abstract method to compute the reward based on the observation and action.

        :param obs: The observation from the environment.
        :param action: The action taken by the agent.
        :return: The computed reward.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def calc_deviation_penalty(self, deviation):
        """
        Calculate the penalty based on the deviation from the track center.
        The penalty is calculated as a percentage of the deviation relative to the half track width.
        If the deviation is below a certain threshold, the penalty is set to zero.
        :param deviation: The deviation from the track center.
        :return: The calculated penalty.
        """
        deviation_in_percentage = abs(deviation) / self.env.calculate_half_track_width(deviation)
        if deviation_in_percentage < self.config_args.get("deviation_penalty_threshold", 0.1):
            deviation_in_percentage = 0

        return -deviation_in_percentage * self.config_args.get("deviation_penalty_coefficient", 1.0)
    
    def calc_rel_heading_penalty(self, rel_heading):
        """
        Calculate the penalty based on the relative heading of the vehicle.
        The penalty is calculated as a percentage of the relative heading.
        If the relative heading is below a certain threshold, the penalty is set to zero.
        s_max is the maximum steering angle of the vehicle.
        :param rel_heading: The relative heading of the vehicle.
        :return: The calculated penalty.
        """
        rel_heading_in_percentage = abs(rel_heading) / self.env.s_max
        # Ensure the relative heading is within the range of 0 to 1
        rel_heading_in_percentage = np.clip(rel_heading_in_percentage, 0, 1)
        if rel_heading_in_percentage < self.config_args.get("rel_heading_penalty_threshold", 0):
            rel_heading_in_percentage = 0

        return -rel_heading_in_percentage * self.config_args.get("rel_heading_penalty_coefficient", 0.25)
    
    def calc_action_smoothness_penalty(self, action):
        """
        Calculate the penalty based on the smoothness of the action.
        The penalty is calculated based on the difference between the current action and the previous action.
        If there are no previous actions, the penalty is set to zero.
        s_max is the maximum steering angle of the vehicle.
        v_max is the maximum speed of the vehicle.
        :param action: The current action taken by the agent.
        :return: The calculated penalty.
        """
        try:
            prev_steering = self.env.last_actions[0][0]
            prev_speed = self.env.last_actions[0][1]
        except TypeError: 
            logging.error("No previous actions found. Returning 0 as penalty. This happens at the first iteration")
            return 0
        except AttributeError:
            logging.error("No previous actions found. Returning 0 as penalty.")
            return 0
        
        steer_delta = abs(prev_steering - action[0, 0])/self.env.s_max
        speed_delta = abs(prev_speed - action[0, 1])/self.env.v_max
        
        steer_penalty = steer_delta * self.config_args.get("steer_smoothness_penalty_coefficient", 1.0)
        speed_penalty = speed_delta * self.config_args.get("speed_smoothness_penalty_coefficient", 1.0)
        
        return steer_penalty + speed_penalty

    def get_reward_adv(self):
        """
        Returns the advancement of last timestep calculated along the central line
        """
        adv = self.env.frenet_s - self.env.prev_frenet_s
        max_adv = self.env.v_max * self.env.timestep
        adv_percentage = adv / max_adv
        return adv_percentage
    
    def get_reward_speed(self):
        """
        Returns the speed of the vehicle as a reward.
        The speed is normalized by the maximum speed of the vehicle.
        """
        speed = self.env.sim.agents[self.env.ego_idx].state[3]
        return speed / self.env.v_max
    
    def get_reward_collision(self):
        """
        Returns a penalty for collision.
        If the vehicle is in collision, the penalty is -1.0, otherwise 0.0.
        """
        if self.env.sim.agents[self.env.ego_idx].collision:
            return -1.0
        return 0.0
    
    def get_reward_tire_slip(self):
        """
        Returns a penalty for tire slip.
        The penalty is calculated based on the lateral velocity of the vehicle.
        If the lateral velocity is above a certain threshold, a penalty is applied.
        :return: The calculated penalty for tire slip.
        """
        penalty = 0
        if abs(self.env.sim.agents[self.env.ego_idx].state[6]) > 0.5:
            penalty = -self.env.sim.agents[self.env.ego_idx].state[6] * 10

        return penalty
    
    def get_reward_lap_time(self):
        """
        Returns the lap time as a reward.
        The lap time is normalized by the maximum lap time.
        If the lap time is zero, the reward is set to 0.0.
        :return: The normalized lap time.
        """
        lap_time = self.env.sim.agents[self.env.ego_idx].lap_time
        if lap_time == 0:
            return 0.0
        return lap_time / self.env.max_lap_time
    

    

    

    

    
