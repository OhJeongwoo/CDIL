# CDIL

https://github.com/openai/mujoco-py/blob/master/mujoco_py/pxd/mjdata.pxd

## environment customization

In this explanation, we assume you only use Mujoco for testing reinforcement learning algorithms. You need to do the followings for customizing your environment.

1. add asset file to /envs/mujoco/assets/

This file describes your physical agent or all of things in your environment such as obstacles or walls. You can define custom model via mujoco XML format.

2. add environment code to /envs/mujoco/

This python code is constructed by typical python gym code format. It needs 'init', 'step', 'reset' functions. Please refer other environment codes for writing your custion environment script.

3. register your environment

You can import your custom code with the follow code.

```
env = gym.make({YOUR_ENV_NAME})
```

Before importing the environment, you need to register the environment. Please open /envs/__init__.py code. You can see the registered environment in this code. You register your environment similar to the others, then it works. 