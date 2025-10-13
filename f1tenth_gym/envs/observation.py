from __future__ import annotations
from abc import abstractmethod
from typing import List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..envs import track
from tf_transformations import euler_from_quaternion
from ..envs import normalization


# copy this function to anywhere you want
def transform_point_to_car_frame(self, point, car_state):
    # Transform the point to the car's frame
    x = point[0] - car_state[0]
    y = point[1] - car_state[1]
    # Rotate the point to the car's frame
    # yaw = euler_from_quaternion([car_state[3],
    #                                 car_state[4],
    #                                 car_state[5],
    #                                 car_state[6]])[2]
    yaw = car_state[4]
    x_car = x * np.cos(yaw) + y * np.sin(yaw)
    y_car = -x * np.sin(yaw) + y * np.cos(yaw)
    return np.array([x_car, y_car])


class Observation:
    """
    Abstract class for observations. Each observation must implement the space and observe methods.

    :param env: The environment.
    :param vehicle_id: The id of the observer vehicle.
    :param kwargs: Additional arguments.
    """

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def space(self):
        raise NotImplementedError()

    @abstractmethod
    def observe(self):
        raise NotImplementedError()

class DirectObservation(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.unwrapped.sim.agents[0].scan_simulator.max_range
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)
        
        complete_space = {}
        for agent_id in self.env.unwrapped.agent_ids:
            agent_dict = {
                "scan": gym.spaces.Box(
                    low=0.0, high=scan_range, shape=(scan_size,), dtype=np.float32
                ),
                "std_state": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(7,), dtype=np.float32
                ),
                "state": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(int(self.env.model.state_dim),), dtype=np.float32
                ),
                # "frenet_state": gym.spaces.Box(
                #     low=-large_num, high=large_num, shape=(7,), dtype=np.float32
                # ),
                "collision": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=np.float32
                ),
                "lap_time": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
                "lap_count": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
                "sim_time": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
            }
            complete_space[agent_id] = gym.spaces.Dict(
                agent_dict
            )

        obs_space = gym.spaces.Dict(complete_space)
        return obs_space

    def observe(self):
        obs = {}  # dictionary agent_id -> observation dict

        for i, agent_id in enumerate(self.env.unwrapped.agent_ids):
            scan = self.env.unwrapped.sim.agent_scans[i]
            agent = self.env.unwrapped.sim.agents[i]

            # create agent's observation dict
            agent_obs = {
                "scan": scan,
                "std_state": agent.standard_state,
                "state": agent.state,
                "collision": agent.in_collision,
                "lap_time": self.env.unwrapped.lap_times[i],
                "lap_count": self.env.unwrapped.lap_counts[i],
                "sim_time": self.env.unwrapped.sim_time,
            }

            # add agent's observation to multi-agent observation
            obs[agent_id] = agent_obs
            
            # cast to match observation space
            for key in obs[agent_id].keys():
                obs[agent_id][key] = np.array(obs[agent_id][key], dtype=np.float32)
        return obs

class OriginalObservation(Observation):
    def __init__(self, env):
        super().__init__(env)

    def space(self):
        num_agents = self.env.unwrapped.num_agents
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = (
            self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5
        )  # add 1.0 to avoid small errors
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)
        obs_space = gym.spaces.Dict(
            {
                "ego_idx": gym.spaces.Discrete(num_agents),
                "scans": gym.spaces.Box(
                    low=0.0,
                    high=scan_range,
                    shape=(num_agents, scan_size),
                    dtype=np.float32,
                ),
                "poses_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_theta": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "ang_vels_z": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "collisions": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32
                ),
                "lap_times": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32
                ),
                "lap_counts": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32
                ),
                "sim_time": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
            }
        )

        return obs_space

    def observe(self):
        # state indices
        xi, yi, deltai, vxi, yawi, yaw_ratei, slipi = range(
            7
        )  # 7 largest state size (ST Model)

        observations = {
            "ego_idx": self.env.unwrapped.sim.ego_idx,
            "scans": [],
            "poses_x": [],
            "poses_y": [],
            "poses_theta": [],
            "linear_vels_x": [],
            "linear_vels_y": [],
            "ang_vels_z": [],
            "collisions": [],
            "lap_times": [],
            "lap_counts": [],
            "sim_time": [],
        }

        for i, agent in enumerate(self.env.unwrapped.sim.agents):
            agent_scan = self.env.unwrapped.sim.agent_scans[i]
            lap_time = self.env.unwrapped.lap_times[i]
            lap_count = self.env.unwrapped.lap_counts[i]
            collision = self.env.unwrapped.sim.collisions[i]

            std_state = agent.standard_state
            x, y, theta = std_state[0], std_state[1], std_state[4]
            vx = std_state[3] * np.cos(std_state[6])
            vy = std_state[3] * np.sin(std_state[6])
            angvel = std_state[5]

            observations["scans"].append(agent_scan)
            observations["poses_x"].append(x)
            observations["poses_y"].append(y)
            observations["poses_theta"].append(theta)
            observations["linear_vels_x"].append(vx)
            observations["linear_vels_y"].append(vy)
            observations["ang_vels_z"].append(angvel)
            observations["collisions"].append(collision)
            observations["lap_times"].append(lap_time)
            observations["lap_counts"].append(lap_count)
            observations["sim_time"].append(self.env.unwrapped.sim_time)

        # cast to match observation space
        for key in observations.keys():
            if isinstance(observations[key], np.ndarray) or isinstance(
                observations[key], list
            ):
                observations[key] = np.array(observations[key], dtype=np.float32)

        return observations

class TrajBasedObservation(Observation):
    """
    Observation class for the  F1/10th Gym environment that provides trajectory-based observations for RL agents.
    This class extends the base Observation class and implements the space and observe methods to provide a structured observation space and observation data for each agent in the environment.
    """

    def __init__(self, env):
        super().__init__(env)
        self.config_args = env.config
        self.track = track.Track.from_track_name(self.config_args["track_name"])

    def space(self):
        num_agents = self.env.unwrapped.num_agents
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = (
            self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5
        )  # add 1.0 to avoid small errors
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)
        obsdim = (self.config_args["scans_num_sectors"] + self.config_args["traj_len"] + 8)*num_agents  # lidar_num + traj + 8 state features
        return spaces.Box(-np.ones(obsdim, dtype=np.float32), np.ones(obsdim, dtype=np.float32))     

    def observe(self):

        observation={}

        for i, agent in enumerate(self.env.unwrapped.sim.agents):
            agent = self.env.unwrapped.sim.agents[i]
            scans = self.env.unwrapped.sim.agent_scans[i]

            std_state = agent.standard_state
            progress_along_track, deviation, rel_heading = self.track.cartesian_to_frenet(
                std_state[0], std_state[1], std_state[4], use_raceline=True)
            closest_point_on_traj = self.track.get_closest_index_on_trajectory(
                std_state[0], std_state[1], use_raceline=True)
            car_state = self.env.unwrapped.sim.agents[i].state
            trajectory = self.track.get_ref_trajectory(closest_point_on_traj, self.config_args["traj_len"], use_raceline=True)
            trajectory_car_frame = np.array([
                transform_point_to_car_frame(self, point, car_state) for point in trajectory
             ]).flatten()
            longitudinal_vel = std_state[3] * np.cos(std_state[6])
            later_vel = std_state[3] * np.sin(std_state[6])
            yaw_rate = std_state[5]


            agent_obs ={
                "scans": scans,
                "traj_car_frame": trajectory_car_frame,
                "progress_along_track": progress_along_track,
                "deviation": deviation,
                "rel_heading": rel_heading,
                "longitudinal_vel": longitudinal_vel,
                "later_vel": later_vel,
                "yaw_rate": yaw_rate,
                "collision": agent.in_collision,
                "lap_time": self.env.unwrapped.lap_times[i],
                "lap_count": self.env.unwrapped.lap_counts[i],
                "sim_time": self.env.unwrapped.sim_time,
                "timestep": self.config_args["timestep"],
            }
        
        observation[self.env.agent_ids[i]] = agent_obs

         # cast to match observation space
        for key in observation.keys():
            if isinstance(observation[key], np.ndarray) or isinstance(
                observation[key], list
            ):
                observation[key] = np.array(observation[key], dtype=np.float32)

        return observation
    
    def aggregate_lidar_scans(self, scans, num_sectors=30):
        sector_size = len(scans) // num_sectors
        aggregated_scans = [
            np.min(scans[i * sector_size : (i + 1) * sector_size])
            for i in range(num_sectors)
        ]
        return np.array(aggregated_scans, dtype=np.float32)
    
    def vectorize_obs(self, observation_dict, norm_action):
        # keys = ['scans', 'traj_car_frame','progress_along_track' 'deviation', 'rel_heading', 'longitudinal_vel', 'later_vel', 'yaw_rate']
        #TODO: Handle multi-agent case
        # for i in range(self.env.unwrapped.num_agents):
        #     if self.env.unwrapped.sim.ego_idx == i:
        #         print(observation_dict.keys())
        #         observation_dict = observation_dict[self.env.agent_ids[i]]
        #         break

        # keys = ['scans', 'traj_car_frame', 'deviation', 'rel_heading', 'longitudinal_vel', 'later_vel', 'yaw_rate', "timestep", 'norm_action']
        keys = ['scans', 'deviation', 'rel_heading', 'longitudinal_vel', 'later_vel', 'yaw_rate', "timestep", 'norm_action']
        scans = observation_dict['scans']
        scans = self.aggregate_lidar_scans(scans, num_sectors=self.config_args["scans_num_sectors"])
        observation_dict['scans'] = np.array(scans, dtype=np.float32)
        observation_dict_norm = normalization.normalise_observation(observation_dict, self.config_args['params'], with_lidar=True)
        # observation_dict_norm['traj_car_frame'] = normalization.normalise_trajectory(observation_dict_norm['traj_car_frame'], self.config_args['traj_len'])

        vectorized_obs = []
        for key in keys:
            if key == 'norm_action':
                vectorized_obs.append(norm_action.flatten())
            else:
                vectorized_obs.append(observation_dict_norm[key].flatten())
            print(key, vectorized_obs[-1].shape)
        vectorized_obs = np.concatenate(vectorized_obs, axis=0)
        vectorized_obs = np.array(vectorized_obs, dtype=np.float32)
        return vectorized_obs
        

def observation_factory(env, type: str | None) -> Observation:
    type = type or "original"

    if type == "original":
        return OriginalObservation(env)
    elif type == "direct":
        return DirectObservation(env)
    elif type == "traj_based":
        return TrajBasedObservation(env)
    else:
        raise ValueError(f"Invalid observation type {type}.")
    
