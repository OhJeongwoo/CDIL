from gym.envs.registration import registry, register, make, spec
from importlib_metadata import entry_points

# Mujoco
# ----------------------------------------

# 2D

register(
    id='Reacher-v2',
    entry_point='envs.mujoco:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Pusher-v2',
    entry_point='envs.mujoco:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Thrower-v2',
    entry_point='envs.mujoco:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Striker-v2',
    entry_point='envs.mujoco:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='InvertedPendulum-v2',
    entry_point='envs.mujoco:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulum-v2',
    entry_point='envs.mujoco:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id='HalfCheetah-v2',
    entry_point='envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HalfCheetah-v3',
    entry_point='envs.mujoco.half_cheetah_v3:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Hopper-v2',
    entry_point='envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Hopper-v3',
    entry_point='envs.mujoco.hopper_v3:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Swimmer-v2',
    entry_point='envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Swimmer-v3',
    entry_point='envs.mujoco.swimmer_v3:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Walker2d-v2',
    max_episode_steps=1000,
    entry_point='envs.mujoco:Walker2dEnv',
)

register(
    id='Walker2d-v3',
    max_episode_steps=1000,
    entry_point='envs.mujoco.walker2d_v3:Walker2dEnv',
)

register(
    id='Ant-v2',
    entry_point='envs.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Ant-v3',
    entry_point='envs.mujoco.ant_v3:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v2',
    entry_point='envs.mujoco:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='Humanoid-v3',
    entry_point='envs.mujoco.humanoid_v3:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidStandup-v2',
    entry_point='envs.mujoco:HumanoidStandupEnv',
    max_episode_steps=1000,
)

# Ant v1 (Original)
register(id='Antv1_1-v0', entry_point='envs.mujoco.ant:Antv1_1', max_episode_steps=1000)
register(id='Antv1_2-v0', entry_point='envs.mujoco.ant:Antv1_2', max_episode_steps=1000)
register(id='Antv1_3-v0', entry_point='envs.mujoco.ant:Antv1_3', max_episode_steps=1000)
register(id='Antv1_4-v0', entry_point='envs.mujoco.ant:Antv1_4', max_episode_steps=1000)
register(id='Antv1_5-v0', entry_point='envs.mujoco.ant:Antv1_5', max_episode_steps=1000)
register(id='Antv1_6-v0', entry_point='envs.mujoco.ant:Antv1_6', max_episode_steps=1000)
register(id='Antv1_7-v0', entry_point='envs.mujoco.ant:Antv1_7', max_episode_steps=1000)
register(id='Antv1_8-v0', entry_point='envs.mujoco.ant:Antv1_8', max_episode_steps=1000)
register(id='Antv1_9-v0', entry_point='envs.mujoco.ant:Antv1_9', max_episode_steps=1000)
register(id='Antv1_10-v0', entry_point='envs.mujoco.ant:Antv1_10', max_episode_steps=1000)
register(id='Antv1_11-v0', entry_point='envs.mujoco.ant:Antv1_11', max_episode_steps=1000)
register(id='Antv1_12-v0', entry_point='envs.mujoco.ant:Antv1_12', max_episode_steps=1000)

register(id='Antv1_alignment-v0', entry_point='envs.mujoco.ant:Antv1_alignment', max_episode_steps=1000)
register(id='Antv1_target-v0', entry_point='envs.mujoco.ant:Antv1_target', max_episode_steps=1000)


# Ant v2 (6legged)
register(id='Antv2_1-v0', entry_point='envs.mujoco.ant_6legged:Antv2_1', max_episode_steps=1000)
register(id='Antv2_2-v0', entry_point='envs.mujoco.ant_6legged:Antv2_2', max_episode_steps=1000)
register(id='Antv2_3-v0', entry_point='envs.mujoco.ant_6legged:Antv2_3', max_episode_steps=1000)
register(id='Antv2_4-v0', entry_point='envs.mujoco.ant_6legged:Antv2_4', max_episode_steps=1000)
register(id='Antv2_5-v0', entry_point='envs.mujoco.ant_6legged:Antv2_5', max_episode_steps=1000)
register(id='Antv2_6-v0', entry_point='envs.mujoco.ant_6legged:Antv2_6', max_episode_steps=1000)
register(id='Antv2_7-v0', entry_point='envs.mujoco.ant_6legged:Antv2_7', max_episode_steps=1000)
register(id='Antv2_8-v0', entry_point='envs.mujoco.ant_6legged:Antv2_8', max_episode_steps=1000)
register(id='Antv2_9-v0', entry_point='envs.mujoco.ant_6legged:Antv2_9', max_episode_steps=1000)
register(id='Antv2_10-v0', entry_point='envs.mujoco.ant_6legged:Antv2_10', max_episode_steps=1000)
register(id='Antv2_11-v0', entry_point='envs.mujoco.ant_6legged:Antv2_11', max_episode_steps=1000)
register(id='Antv2_12-v0', entry_point='envs.mujoco.ant_6legged:Antv2_12', max_episode_steps=1000)

register(id='Antv2_alignment-v0', entry_point='envs.mujoco.ant_6legged:Antv2_alignment', max_episode_steps=1000)
register(id='Antv2_target-v0', entry_point='envs.mujoco.ant_6legged:Antv2_target', max_episode_steps=1000)


# Ant v3 (long leg)
register(id='Antv3_1-v0', entry_point='envs.mujoco.ant_long:Antv3_1', max_episode_steps=1000)
register(id='Antv3_2-v0', entry_point='envs.mujoco.ant_long:Antv3_2', max_episode_steps=1000)
register(id='Antv3_3-v0', entry_point='envs.mujoco.ant_long:Antv3_3', max_episode_steps=1000)
register(id='Antv3_4-v0', entry_point='envs.mujoco.ant_long:Antv3_4', max_episode_steps=1000)
register(id='Antv3_5-v0', entry_point='envs.mujoco.ant_long:Antv3_5', max_episode_steps=1000)
register(id='Antv3_6-v0', entry_point='envs.mujoco.ant_long:Antv3_6', max_episode_steps=1000)
register(id='Antv3_7-v0', entry_point='envs.mujoco.ant_long:Antv3_7', max_episode_steps=1000)
register(id='Antv3_8-v0', entry_point='envs.mujoco.ant_long:Antv3_8', max_episode_steps=1000)
register(id='Antv3_9-v0', entry_point='envs.mujoco.ant_long:Antv3_9', max_episode_steps=1000)
register(id='Antv3_10-v0', entry_point='envs.mujoco.ant_long:Antv3_10', max_episode_steps=1000)
register(id='Antv3_11-v0', entry_point='envs.mujoco.ant_long:Antv3_11', max_episode_steps=1000)
register(id='Antv3_12-v0', entry_point='envs.mujoco.ant_long:Antv3_12', max_episode_steps=1000)

register(id='Antv4_1-v0', entry_point='envs.mujoco.ant_reacher:Antv4_1', max_episode_steps=200)
register(id='Antv4_alignment-v0', entry_point='envs.mujoco.ant_reacher:Antv4_alignment', max_episode_steps=200)
register(id='Antv4_target-v0', entry_point='envs.mujoco.ant_reacher:Antv4_target', max_episode_steps=200)

register(id='Antv5_1-v0', entry_point='envs.mujoco.ant_6legged_reacher:Antv5_1', max_episode_steps=200)
register(id='Antv5_alignment-v0', entry_point='envs.mujoco.ant_6legged_reacher:Antv5_alignment', max_episode_steps=200)
register(id='Antv5_target-v0', entry_point='envs.mujoco.ant_6legged_reacher:Antv5_target', max_episode_steps=200)

register(id='Antv6_alignment-v0', entry_point='envs.mujoco.ant_reacher_slow:Antv6_alignment', max_episode_steps=400)
register(id='Antv6_target-v0', entry_point='envs.mujoco.ant_reacher_slow:Antv6_target', max_episode_steps=400)

register(id='Antv7_alignment-v0', entry_point='envs.mujoco.ant_6legged_reacher_slow:Antv7_alignment', max_episode_steps=400)
register(id='Antv7_target-v0', entry_point='envs.mujoco.ant_6legged_reacher_slow:Antv7_target', max_episode_steps=400)

register(id='Antv8_alignment-v0', entry_point='envs.mujoco.ant_reacher_fast:Antv8_alignment', max_episode_steps=100)
register(id='Antv8_target-v0', entry_point='envs.mujoco.ant_reacher_fast:Antv8_target', max_episode_steps=100)

register(id='Antv9_alignment-v0', entry_point='envs.mujoco.ant_6legged_reacher_fast:Antv9_alignment', max_episode_steps=100)
register(id='Antv9_target-v0', entry_point='envs.mujoco.ant_6legged_reacher_fast:Antv9_target', max_episode_steps=100)

register(id='Antv3_alignment-v0', entry_point='envs.mujoco.ant_long:Antv3_alignment', max_episode_steps=1000)
register(id='Antv3_target-v0', entry_point='envs.mujoco.ant_long:Antv3_target', max_episode_steps=1000)

register(id='Reacher2DOF-v0', entry_point='envs.mujoco.reacher_2dof:Reacher2DOFEnv', max_episode_steps=60)
register(id='Reacher2DOFCorner-v0', entry_point='envs.mujoco.reacher_2dof:Reacher2DOFCornerEnv', max_episode_steps=60)
register(id='Reacher2DOFWall-v0', entry_point='envs.mujoco.reacher_2dof:Reacher2DOFWallEnv', max_episode_steps=60)

register(id='Reacher2DOFDynamicsCorner-v0', entry_point='envs.mujoco.reacher_2dof_dynamics:Reacher2DOFDynamicsCornerEnv', max_episode_steps=60)
register(id='Reacher2DOFDynamicsWall-v0', entry_point='envs.mujoco.reacher_2dof_dynamics:Reacher2DOFDynamicsWallEnv', max_episode_steps=60)

register(id='Reacher2DOFViewpointCorner-v0', entry_point='envs.mujoco.reacher_2dof_viewpoint:Reacher2DOFViewpointCornerEnv', max_episode_steps=60)
register(id='Reacher2DOFViewpointWall-v0', entry_point='envs.mujoco.reacher_2dof_viewpoint:Reacher2DOFViewpointWallEnv', max_episode_steps=60)


register(id='Reacher3DOF-v0', entry_point='envs.mujoco.reacher_3dof:Reacher3DOFEnv', max_episode_steps=60)
register(id='Reacher3DOFCorner-v0', entry_point='envs.mujoco.reacher_3dof:Reacher3DOFCornerEnv', max_episode_steps=60)
register(id='Reacher3DOFWall-v0', entry_point='envs.mujoco.reacher_3dof:Reacher3DOFWallEnv', max_episode_steps=60)

register(id='Reacher2DOFVerySlowCorner-v0', entry_point='envs.mujoco.reacher_2dof_very_slow:Reacher2DOFVerySlowCornerEnv', max_episode_steps=240)
register(id='Reacher2DOFVerySlowWall-v0', entry_point='envs.mujoco.reacher_2dof_very_slow:Reacher2DOFVerySlowWallEnv', max_episode_steps=240)

register(id='Reacher2DOFSlowCorner-v0', entry_point='envs.mujoco.reacher_2dof_slow:Reacher2DOFSlowCornerEnv', max_episode_steps=120)
register(id='Reacher2DOFSlowWall-v0', entry_point='envs.mujoco.reacher_2dof_slow:Reacher2DOFSlowWallEnv', max_episode_steps=120)

register(id='Reacher2DOFLittleSlowCorner-v0', entry_point='envs.mujoco.reacher_2dof_little_slow:Reacher2DOFLittleSlowCornerEnv', max_episode_steps=80)
register(id='Reacher2DOFLittleSlowWall-v0', entry_point='envs.mujoco.reacher_2dof_little_slow:Reacher2DOFLittleSlowWallEnv', max_episode_steps=80)

register(id='Reacher2DOFLittleFastCorner-v0', entry_point='envs.mujoco.reacher_2dof_little_fast:Reacher2DOFLittleFastCornerEnv', max_episode_steps=40)
register(id='Reacher2DOFLittleFastWall-v0', entry_point='envs.mujoco.reacher_2dof_little_fast:Reacher2DOFLittleFastWallEnv', max_episode_steps=40)

register(id='Reacher2DOFFastCorner-v0', entry_point='envs.mujoco.reacher_2dof_fast:Reacher2DOFFastCornerEnv', max_episode_steps=30)
register(id='Reacher2DOFFastWall-v0', entry_point='envs.mujoco.reacher_2dof_fast:Reacher2DOFFastWallEnv', max_episode_steps=30)

register(id='Reacher2DOFVeryFastCorner-v0', entry_point='envs.mujoco.reacher_2dof_very_fast:Reacher2DOFVeryFastCornerEnv', max_episode_steps=15)
register(id='Reacher2DOFVeryFastWall-v0', entry_point='envs.mujoco.reacher_2dof_very_fast:Reacher2DOFVeryFastWallEnv', max_episode_steps=15)

register(id='Pusher3DOFCorner-v0', entry_point='envs.mujoco.pusher_3dof:Pusher3DOFCornerEnv', max_episode_steps=500)
register(id='Pusher3DOFWall-v0', entry_point='envs.mujoco.pusher_3dof:Pusher3DOFWallEnv', max_episode_steps=500)

register(id='Pusher2DOFCorner-v0', entry_point='envs.mujoco.pusher_2dof:Pusher2DOFCornerEnv', max_episode_steps=500)
register(id='Pusher2DOFWall-v0', entry_point='envs.mujoco.pusher_2dof:Pusher2DOFWallEnv', max_episode_steps=500)

register(id='Pusher2DOFVerySlowCorner-v0', entry_point='envs.mujoco.pusher_2dof_very_slow:Pusher2DOFVerySlowCornerEnv', max_episode_steps=2000)
register(id='Pusher2DOFVerySlowWall-v0', entry_point='envs.mujoco.pusher_2dof_slow_slow:Pusher2DOFVerySlowWallEnv', max_episode_steps=2000)

register(id='Pusher2DOFSlowCorner-v0', entry_point='envs.mujoco.pusher_2dof_slow:Pusher2DOFSlowCornerEnv', max_episode_steps=1000)
register(id='Pusher2DOFSlowWall-v0', entry_point='envs.mujoco.pusher_2dof_slow:Pusher2DOFSlowWallEnv', max_episode_steps=1000)

register(id='Pusher2DOFFastCorner-v0', entry_point='envs.mujoco.pusher_2dof_fast:Pusher2DOFFastCornerEnv', max_episode_steps=250)
register(id='Pusher2DOFFastWall-v0', entry_point='envs.mujoco.pusher_2dof_fast:Pusher2DOFFastWallEnv', max_episode_steps=250)

register(id='Pusher2DOFVeryFastCorner-v0', entry_point='envs.mujoco.pusher_2dof_very_fast:Pusher2DOFVeryFastCornerEnv', max_episode_steps=125)
register(id='Pusher2DOFVeryFastWall-v0', entry_point='envs.mujoco.pusher_2dof_very_fast:Pusher2DOFVeryFastWallEnv', max_episode_steps=125)

register(id='Swimmer_alignment-v0', entry_point='envs.mujoco.swimmer_reacher:Swimmer_alignment', max_episode_steps=1000)
register(id='Swimmer_target-v0', entry_point='envs.mujoco.swimmer_reacher:Swimmer_target', max_episode_steps=1000)

