import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math
import random


class Antv2_1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 10.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        print("hi")
        print(self.sim.data.qpos.shape)
        print(self.sim.data.qvel.shape)
        print(self.sim.data.cfrc_ext.shape)
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -10.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 80.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 100.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 170.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_6(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -170.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_7(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -80.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_8(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -100.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_9(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 45.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_10(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 135.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_11(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -45.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_12(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -135.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_alignment(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta_list = [10, -10, 80, 100, -80, -100, 170, -170]
        self.N = len(self.theta_list)
        self.theta = 0.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                [self.ct, self.st],
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0
        idx = random.randint(0, self.N - 1)
        self.theta = self.theta_list[idx] / 180.0 * math.pi
        self.ct = math.cos(self.theta)
        self.st = math.sin(self.theta)
        print("[%d] direction %.2f" %(self.i_episode, self.theta))

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class Antv2_target(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta_list = [45, 135, -45, -135]
        self.N = len(self.theta_list)
        self.theta = 0.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant_6legged.xml", 5)
        utils.EzPickle.__init__(self)        

    def step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                [self.ct, self.st],
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0
        idx = random.randint(0, self.N - 1)
        self.theta = self.theta_list[idx] / 180.0 * math.pi
        self.ct = math.cos(self.theta)
        self.st = math.sin(self.theta)
        print("[%d] direction %.2f" %(self.i_episode, self.theta))

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
