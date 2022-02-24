import os
import numpy as np
import gym
import pickle
import torch

def sample_demo(X, n):
    N = len(X['obs'])
    rt = np.random.randint(N, size= n)
    return X['obs'][rt], X['act'][rt], X['nobs'][rt]

def sample_batch(X, n):
    return np.random.choice(X, n)


def get_env_dim(env_name):
    try:
        env = gym.make(env_name)
        return env.observation_space.shape[0], env.action_space.shape[0]
    except:
        print("Failed to check environment info")
        return 0, 0


def load_pickle(file_name):
    with open(file_name, 'rb') as pk:
        data = pickle.load(pk)
    return data

def load_demo(file_name):
    data = load_pickle(file_name)
    obs = data['obs']
    act = data['act']
    nobs = data['nobs']
    rt_obs = []
    rt_act = []
    rt_nobs = []
    n_epi = len(obs)
    for i_epi in range(n_epi):
        epi_len = len(obs[i_epi])
        for i in range(epi_len):
            rt_obs.append(obs[i_epi][i])
            rt_act.append(act[i_epi][i])
            rt_nobs.append(nobs[i_epi][i])
    rt_obs = np.array(rt_obs)
    rt_act = np.array(rt_act)
    rt_nobs = np.array(rt_nobs)
    return {'obs': rt_obs, 'act': rt_act, 'nobs': rt_nobs}

def check_path(path_name):
    if not os.path.exists(path_name):
        os.mkdir(path_name)

def separate_goal(X, offset, dim):
    X_goal = X[:, offset:offset+dim]
    X = np.concatenate((X[:,:offset], X[:, offset+dim:]), axis=1)
    return X_goal, X

def tensor_from_numpy(X, device_):
    return torch.from_numpy(X).type(torch.FloatTensor).to(device=device_)