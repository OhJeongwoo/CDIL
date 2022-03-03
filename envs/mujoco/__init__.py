from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv

from gym.envs.mujoco.reacher_3dof import Reacher3DOFEnv, Reacher3DOFCornerEnv, Reacher3DOFWallEnv
from gym.envs.mujoco.reacher_2dof import Reacher2DOFEnv, Reacher2DOFCornerEnv, Reacher2DOFWallEnv

from gym.envs.mujoco.ant import Antv1_1
from gym.envs.mujoco.ant import Antv1_2
from gym.envs.mujoco.ant import Antv1_3
from gym.envs.mujoco.ant import Antv1_4
from gym.envs.mujoco.ant import Antv1_5
from gym.envs.mujoco.ant import Antv1_6
from gym.envs.mujoco.ant import Antv1_7
from gym.envs.mujoco.ant import Antv1_8
from gym.envs.mujoco.ant import Antv1_9
from gym.envs.mujoco.ant import Antv1_10
from gym.envs.mujoco.ant import Antv1_11
from gym.envs.mujoco.ant import Antv1_12

from gym.envs.mujoco.ant import Antv1_alignment
from gym.envs.mujoco.ant import Antv1_target

from gym.envs.mujoco.ant_6legged import Antv2_1
from gym.envs.mujoco.ant_6legged import Antv2_2
from gym.envs.mujoco.ant_6legged import Antv2_3
from gym.envs.mujoco.ant_6legged import Antv2_4
from gym.envs.mujoco.ant_6legged import Antv2_5
from gym.envs.mujoco.ant_6legged import Antv2_6
from gym.envs.mujoco.ant_6legged import Antv2_7
from gym.envs.mujoco.ant_6legged import Antv2_8
from gym.envs.mujoco.ant_6legged import Antv2_9
from gym.envs.mujoco.ant_6legged import Antv2_10
from gym.envs.mujoco.ant_6legged import Antv2_11
from gym.envs.mujoco.ant_6legged import Antv2_12

from gym.envs.mujoco.ant_6legged import Antv2_alignment
from gym.envs.mujoco.ant_6legged import Antv2_target

from gym.envs.mujoco.ant_long import Antv3_1
from gym.envs.mujoco.ant_long import Antv3_2
from gym.envs.mujoco.ant_long import Antv3_3
from gym.envs.mujoco.ant_long import Antv3_4
from gym.envs.mujoco.ant_long import Antv3_5
from gym.envs.mujoco.ant_long import Antv3_6
from gym.envs.mujoco.ant_long import Antv3_7
from gym.envs.mujoco.ant_long import Antv3_8
from gym.envs.mujoco.ant_long import Antv3_9
from gym.envs.mujoco.ant_long import Antv3_10
from gym.envs.mujoco.ant_long import Antv3_11
from gym.envs.mujoco.ant_long import Antv3_12

from gym.envs.mujoco.ant_reacher import Antv4_1
from gym.envs.mujoco.ant_reacher import Antv4_alignment, Antv4_target

from gym.envs.mujoco.ant_6legged_reacher import Antv5_1
from gym.envs.mujoco.ant_6legged_reacher import Antv5_alignment, Antv5_target

from gym.envs.mujoco.ant_reacher_slow import Antv6_alignment, Antv6_target
from gym.envs.mujoco.ant_6legged_reacher_slow import Antv7_alignment, Antv7_target
from gym.envs.mujoco.ant_reacher_fast import Antv8_alignment, Antv8_target
from gym.envs.mujoco.ant_6legged_reacher_fast import Antv9_alignment, Antv9_target

from gym.envs.mujoco.ant_long import Antv3_alignment
from gym.envs.mujoco.ant_long import Antv3_target

from gym.envs.mujoco.reacher_2dof_very_slow import Reacher2DOFVerySlowCornerEnv, Reacher2DOFVerySlowWallEnv
from gym.envs.mujoco.reacher_2dof_slow import Reacher2DOFSlowCornerEnv, Reacher2DOFSlowWallEnv
from gym.envs.mujoco.reacher_2dof_fast import Reacher2DOFFastCornerEnv, Reacher2DOFFastWallEnv
from gym.envs.mujoco.reacher_2dof_very_fast import Reacher2DOFVeryFastCornerEnv, Reacher2DOFVeryFastWallEnv

from gym.envs.mujoco.pusher_3dof import Pusher3DOFCornerEnv, Pusher3DOFWallEnv
from gym.envs.mujoco.pusher_2dof import Pusher2DOFCornerEnv, Pusher2DOFWallEnv
from gym.envs.mujoco.pusher_2dof_very_slow import Pusher2DOFVerySlowCornerEnv, Pusher2DOFVerySlowWallEnv
from gym.envs.mujoco.pusher_2dof_slow import Pusher2DOFSlowCornerEnv, Pusher2DOFSlowWallEnv
from gym.envs.mujoco.pusher_2dof_fast import Pusher2DOFFastCornerEnv, Pusher2DOFFastWallEnv
from gym.envs.mujoco.pusher_2dof_very_fast import Pusher2DOFVeryFastCornerEnv, Pusher2DOFVeryFastWallEnv

from gym.envs.mujoco.swimmer_reacher import Swimmer_alignment, Swimmer_target
