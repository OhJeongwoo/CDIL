import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
# from mujoco_py.mjlib import mjlib
import random
import pdb

import os

class Pusher2DOFFastCornerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.steps = 0
        self.i_episode = 0
        scale = np.sqrt(2)
        self.det_corner_options = [[-0.25/scale, -0.25/scale], [-0.25/scale, 0.25/scale], [0.25/scale, -0.25/scale], [0.25/scale, 0.25/scale],
                                [-0.2/scale, -0.2/scale], [-0.2/scale, 0.2/scale], [0.2/scale, -0.2/scale], [0.2/scale, 0.2/scale],
                                [-0.15/scale, -0.15/scale], [-0.15/scale, 0.15/scale], [0.15/scale, -0.15/scale], [0.15/scale, 0.15/scale]]
        self.N = len(self.det_corner_options)
        self.reset_called = False
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher_2dof_fast.xml', 2)
        self.viewer = None

    def step(self, a):
        self.steps += 1
        vec = self.get_body_com("target")-self.get_body_com("destination")

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.abs(a).sum()
        a = np.clip(a, -0.002, 0.002)

        proximity_threshold = 0.05
        reward = reward_dist + 100 * int(-reward_dist < proximity_threshold)
        reward = reward * 0.25 * 2.0

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        done = (-reward_dist < proximity_threshold) and self.reset_called

        '''
        if np.linalg.norm(vec) < 0.02:
            done = True
        '''
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:] = [0, 0, 0] # [-0.1, 0, 0]
        self.viewer.cam.elevation = -90 # -60
        self.viewer.cam.distance = 1.1
        self.viewer.cam.azimuth = 0

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0
        self.reset_called = True
        n_joints = 2
        max_reachable_len = (n_joints+1) * 0.1 # .1 is the length of each link
        min_reachable_len = 0.1 # joint ranges: inf, .9, 2.8

        bias_low = 0.8
        bias_high = 0.9
        bias2_low = -2.8
        bias2_high = 2.8
        first_bias = self.np_random.uniform(low=bias_low, high=bias_high, size=1)
        second_bias = self.np_random.uniform(low=bias2_low, high=bias2_high, size=1)
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[:1] = first_bias
        #qpos[1:2] = second_bias
        #print("using super diverse starts")
        while True:
            chosen_goal = self.det_corner_options[random.randrange(self.N)]
            self.goal = np.array(chosen_goal)
#           
            if np.linalg.norm(self.goal) < max_reachable_len and np.linalg.norm(self.goal) > min_reachable_len:
                break
        print("[%d] goal: (%.2f, %.2f)" %(self.i_episode, self.goal[0], self.goal[1]))
        qpos[-2:] = self.goal
        qpos[-4:-2] = self.goal + np.array([0.2, 0.2])
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        n_joints = 2

        theta = self.sim.data.qpos.flat[:n_joints]

        return np.concatenate([
            self.sim.data.qpos.flat[n_joints:n_joints+2], # box position
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:n_joints] # joint velocities
        ])

        #self.get_body_com("fingertip") - self.get_body_com("target")

    def set_state_from_obs(self, obs):
        n_joints = 2
        qvel = np.zeros((self.model.nv, ))

        # Positions
        cos_theta = obs[2:n_joints+2]
        sin_theta = obs[n_joints+2:2*n_joints+2]
        theta = np.arctan2(sin_theta, cos_theta) # 3
        target = obs[0:2] # 2

        qpos = np.concatenate([theta, target, self.goal], axis=0)
        qvel[:n_joints] = obs[2*n_joints+2:2*n_joints+2+n_joints] # 5

        self.set_state(qpos, qvel)

    # def _get_viewer(self):
    #     if self.viewer is None:
    #         size = 128
    #         self.viewer = mujoco_py.MjViewer(visible=True, init_width=size, init_height=size, go_fast=False)
    #         self.viewer.start()
    #         self.viewer.set_model(self.model)
    #         self.viewer_setup()
    #     return self.viewer


class Pusher2DOFFastWallEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.steps = 0
        self.i_episode = 0
        self.det_wall_options = [[0.25, 0], [-0.25, 0], [0, 0.25], [0, -0.25],
                                [0.2, 0], [-0.2, 0], [0, 0.2], [0, -0.2],
                                [0.15, 0], [-0.15, 0], [0, 0.15], [0, -0.15]]
        
        self.N = len(self.det_wall_options)
        self.reset_called = False
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher_2dof_fast.xml', 2)
        self.viewer = None

    def step(self, a):
        self.steps += 1
        vec = self.get_body_com("target")-self.get_body_com("destination")

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.abs(a).sum()
        a = np.clip(a, -0.002, 0.002)

        proximity_threshold = 0.05
        reward = reward_dist + 100 * int(-reward_dist < proximity_threshold)
        reward = reward * 0.25 * 2.0

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        done = (-reward_dist < proximity_threshold) and self.reset_called

        '''
        if np.linalg.norm(vec) < 0.02:
            done = True
        '''
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:] = [0, 0, 0] # [-0.1, 0, 0]
        self.viewer.cam.elevation = -90 # -60
        self.viewer.cam.distance = 1.1
        self.viewer.cam.azimuth = 0

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0
        self.reset_called = True
        n_joints = 2
        max_reachable_len = (n_joints+1) * 0.1 # .1 is the length of each link
        min_reachable_len = 0.1 # joint ranges: inf, .9, 2.8

        bias_low = 0.8
        bias_high = 0.9
        bias2_low = -2.8
        bias2_high = 2.8
        first_bias = self.np_random.uniform(low=bias_low, high=bias_high, size=1)
        second_bias = self.np_random.uniform(low=bias2_low, high=bias2_high, size=1)
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[:1] = first_bias
        #qpos[1:2] = second_bias
        #print("using super diverse starts")
        while True:
            chosen_goal = self.det_wall_options[random.randrange(self.N)]
            self.goal = np.array(chosen_goal)
#           
            if np.linalg.norm(self.goal) < max_reachable_len and np.linalg.norm(self.goal) > min_reachable_len:
                break
        print("[%d] goal: (%.2f, %.2f)" %(self.i_episode, self.goal[0], self.goal[1]))
        qpos[-2:] = self.goal
        qpos[-4:-2] = self.goal + np.array([0.2, 0.2])
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        n_joints = 2

        theta = self.sim.data.qpos.flat[:n_joints]

        return np.concatenate([
            self.sim.data.qpos.flat[n_joints:n_joints+2], # box position
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:n_joints] # joint velocities
        ])

        #self.get_body_com("fingertip") - self.get_body_com("target")

    def set_state_from_obs(self, obs):
        n_joints = 2
        qvel = np.zeros((self.model.nv, ))

        # Positions
        cos_theta = obs[2:n_joints+2]
        sin_theta = obs[n_joints+2:2*n_joints+2]
        theta = np.arctan2(sin_theta, cos_theta) # 3
        target = obs[0:2] # 2

        qpos = np.concatenate([theta, target, self.goal], axis=0)
        qvel[:n_joints] = obs[2*n_joints+2:2*n_joints+2+n_joints] # 5

        self.set_state(qpos, qvel)

    # def _get_viewer(self):
    #     if self.viewer is None:
    #         size = 128
    #         self.viewer = mujoco_py.MjViewer(visible=True, init_width=size, init_height=size, go_fast=False)
    #         self.viewer.start()
    #         self.viewer.set_model(self.model)
    #         self.viewer_setup()
    #     return self.viewer
