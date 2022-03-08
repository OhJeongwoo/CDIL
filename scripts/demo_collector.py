import torch
import yaml
import os
import sys
import gym
import pickle
import argparse

from utils import *

sys.path.append(os.getcwd() + '/..')
import envs

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/yaml/"
DEMO_PATH = PROJECT_PATH + "/demo/"

if __name__ == "__main__":
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Demo Collector')
    parser.add_argument('--yaml', default='test', type=str)
    args = parser.parse_args()

    YAML_FILE = YAML_PATH + args.yaml + ".yaml"

    # set yaml path
    with open(YAML_FILE) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
        print(args)

    # set hyperparameters
    exp_name_ = args['exp_name']
    exp_env_name_ = args['exp_env_name']
    exp_obs_dim_, exp_act_dim_ = get_env_dim(exp_env_name_)
    exp_epi_len_ = args['exp_epi_len']
    env_ = gym.make(exp_env_name_)
    exp_policy_file_name_ = args['exp_policy_file_name']
    exp_demo_file_name_ = args['exp_demo_file_name']
    n_episode_ = args['n_episode']
    policy_file_ = POLICY_PATH + exp_name_ + "/" + exp_policy_file_name_ + ".pt"
    demo_file_ = DEMO_PATH + exp_demo_file_name_ + ".pkl"

    # set seed
    seed_ = args['seed']
    np.random.seed(seed_)

    # set rendering
    render_ = args['render']

    # initialize environment
    env_ = gym.make(exp_env_name_)

    # load saved policy
    model = torch.load(policy_file_)

    def get_action(o, remove_grad=True):
        a = model.act(torch.unsqueeze(torch.as_tensor(o, dtype=torch.float32), dim=0).to(device=device_))
        if remove_grad:
            return torch.squeeze(a.detach().cpu()).numpy()
        return a
    
    o, r, d, ep_ret, ep_len, n = env_.reset(), 0, False, 0, 0, 0
    i_episode = 0
    obs_list = []
    act_list = []
    nobs_list = []
    obs = []
    acts = []
    nobs =[]
    tot_ep_ret = 0
    while i_episode < n_episode_:
        if render_:
            env_.render()
        a = get_action(o)
        no, r, d, _ = env_.step(a)
        ep_ret += r
        ep_len += 1
        obs.append(o)
        acts.append(a)
        nobs.append(no)
        o = no
        if d or (ep_len == exp_epi_len_):
            print("Episode %d \t EpRet %.3f \t EpLen %d" %(i_episode, ep_ret, ep_len))
            i_episode += 1
            tot_ep_ret += ep_ret
            o = env_.reset()
            ep_ret = 0
            ep_len = 0
            obs_list.append(obs)
            act_list.append(acts)
            nobs_list.append(nobs)
            obs = []
            acts = []
            nobs =[]
    data = {'obs': obs_list, 'act': act_list, 'nobs': nobs_list}
    output = open(demo_file_, 'wb')
    pickle.dump(data, output)
    print("Success to save demo")
    print("# of episodes: %d, avg EpRet: %.3f" %(n_episode_, tot_ep_ret / n_episode_))
