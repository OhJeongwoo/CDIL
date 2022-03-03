import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
# from mujoco_py.mjlib import mjlib
import random
import pdb

import os

class Reacher2DOFVeryFastCornerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.steps = 0
        self.i_episode = 0
        scale = np.sqrt(2)
        self.det_corner_options = [[-0.25/scale, -0.25/scale], [-0.25/scale, 0.25/scale], [0.25/scale, -0.25/scale], [0.25/scale, 0.25/scale],
                                [-0.2/scale, -0.2/scale], [-0.2/scale, 0.2/scale], [0.2/scale, -0.2/scale], [0.2/scale, 0.2/scale],
                                [-0.15/scale, -0.15/scale], [-0.15/scale, 0.15/scale], [0.15/scale, -0.15/scale], [0.15/scale, 0.15/scale]]
        self.N = len(self.det_corner_options)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_2dof_very_fast.xml', 2)
        self.viewer = None

    def step(self, a):
        self.steps += 1
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = 4.0 * (reward_dist + reward_ctrl)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        done = False
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
        n_joints = self.model.nq - 2
        max_reachable_len = n_joints * 0.1 # .1 is the length of each link
        min_reachable_len = 0.1 # joint ranges: inf, .9, 2.8

        bias_low = -3.14
        bias_high = 3.14
        bias2_low = -2.8
        bias2_high = 2.8
        first_bias = self.np_random.uniform(low=bias_low, high=bias_high, size=1)
        second_bias = self.np_random.uniform(low=bias2_low, high=bias2_high, size=1)
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[:1] = first_bias
        chosen_goal = self.det_corner_options[random.randrange(self.N)]
        self.goal = np.array(chosen_goal)
#       print("[%d] goal: (%.2f, %.2f)" %(self.i_episode, self.goal[0], self.goal[1]))
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        n_joints = self.model.nq - 2

        theta = self.sim.data.qpos.flat[:n_joints]

        return np.concatenate([
            self.sim.data.qpos.flat[n_joints:], # target position
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:n_joints] # joint velocities
        ])

    def set_state_from_obs(self, obs):
        n_joints = self.model.nq - 2
        qvel = np.zeros((self.model.nv, ))

        # Positions
        target = obs[:2]
        cos_theta = obs[2:2+n_joints]
        sin_theta = obs[2+n_joints:2+2*n_joints]
        theta = np.arctan2(sin_theta, cos_theta)

        qpos = np.concatenate([theta, target], axis=0)
        qvel[:n_joints] = obs[2*n_joints+2:2*n_joints+2+n_joints]

        self.set_state(qpos, qvel)

class Reacher2DOFVeryFastWallEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.steps = 0
        self.i_episode = 0
        scale = np.sqrt(2)
        self.det_wall_options = [[0.25, 0], [-0.25, 0], [0, 0.25], [0, -0.25],
                                [0.2, 0], [-0.2, 0], [0, 0.2], [0, -0.2],
                                [0.15, 0], [-0.15, 0], [0, 0.15], [0, -0.15]]
        self.N = len(self.det_wall_options)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_2dof_very_fast.xml', 2)
        self.viewer = None

    def step(self, a):
        self.steps += 1
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = 4.0 * (reward_dist + reward_ctrl)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        done = False
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
        n_joints = self.model.nq - 2
        max_reachable_len = n_joints * 0.1 # .1 is the length of each link
        min_reachable_len = 0.1 # joint ranges: inf, .9, 2.8

        bias_low = -3.14
        bias_high = 3.14
        bias2_low = -2.8
        bias2_high = 2.8
        first_bias = self.np_random.uniform(low=bias_low, high=bias_high, size=1)
        second_bias = self.np_random.uniform(low=bias2_low, high=bias2_high, size=1)
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[:1] = first_bias
        chosen_goal = self.det_wall_options[random.randrange(self.N)]
        self.goal = np.array(chosen_goal)
#       print("[%d] goal: (%.2f, %.2f)" %(self.i_episode, self.goal[0], self.goal[1]))
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        n_joints = self.model.nq - 2

        theta = self.sim.data.qpos.flat[:n_joints]

        return np.concatenate([
            self.sim.data.qpos.flat[n_joints:], # target position
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:n_joints] # joint velocities
        ])

    def set_state_from_obs(self, obs):
        n_joints = self.model.nq - 2
        qvel = np.zeros((self.model.nv, ))

        # Positions
        target = obs[:2]
        cos_theta = obs[2:2+n_joints]
        sin_theta = obs[2+n_joints:2+2*n_joints]
        theta = np.arctan2(sin_theta, cos_theta)

        qpos = np.concatenate([theta, target], axis=0)
        qvel[:n_joints] = obs[2*n_joints+2:2*n_joints+2+n_joints]

        self.set_state(qpos, qvel)

