import torch
import yaml
import os
import gym

from model import MLP
from torch.nn import MSELoss

from utils import sample_batch, get_env_dim, load_demo, sample_demo, check_path, separate_goal

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/test.yaml"
DEMO_PATH = PROJECT_PATH + "/demo/"
RESULT_PATH = PROJECT_PATH + "/result/"

if __name__ == "__main__":
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # set yaml path
    with open(YAML_PATH) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
        print(args)

    # set hyperparameters
    exp_name_ = args['exp_name']
    result_path_ = RESULT_PATH + exp_name_ + "/"
    check_path(result_path_)

    exp_env_name_ = args['exp_env_name']
    exp_obs_dim_, exp_act_dim_ = get_env_dim(exp_env_name_)
    exp_goal_ = args['exp_goal']
    exp_goal_offset_ = args['exp_goal_offset']
    exp_goal_dim_ = args['exp_goal_dim']

    lea_env_name_ = args['lea_env_name']
    lea_obs_dim_, lea_act_dim_ = get_env_dim(lea_env_name_)
    lea_goal_ = args['lea_goal']
    lea_goal_offset_ = args['lea_goal_offset']
    lea_goal_dim_ = args['lea_goal_dim']
    if exp_goal_dim_ != lea_goal_dim_:
        print("[ERR] goal dimesion is different in two domain... can not run GAMA")
        exit()

    exp_policy_file_name_ = args['exp_policy_file_name']
    exp_policy_file_ = POLICY_PATH + exp_policy_file_name_ + ".pt"

    n_episode_ = args['n_episode']
    max_epi_len_ = args['lea_epi_len']

    # initialize environment
    env_ = gym.make(lea_env_name_)
    exp_obs_dim_, exp_act_dim_ = get_env_dim(exp_env_name_)

    # load saved policy
    pi = torch.load(exp_policy_file_)
    f = torch.load(result_path_ + "f.pt")
    g = torch.load(result_path_ + "g.pt")
    
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            x_goal = x[:,lea_goal_offset_:lea_goal_offset_ + lea_goal_dim_]
            x = torch.cat((x[:,:lea_goal_offset_], x[:,lea_goal_offset_ + lea_goal_dim_:]), 1)
            x = f(x)
            x = torch.cat((x[:,:exp_goal_offset_], x_goal, x[:,exp_goal_offset_:]), 1)
            action = pi(x)
        return action
    
    o, r, d, ep_ret, ep_len, n = env_.reset(), 0, False, 0, 0, 0
    i_episode = 0
    tot_ep_ret = 0
    while i_episode < n_episode_:
        a = get_action(o)
        no, r, d, _ = env_.step(a)
        ep_ret += r
        ep_len += 1
        o = no
        if d or (ep_len == max_epi_len_):
            print("Episode %d \t EpRet %.3f \t EpLen %d" %(i_episode, ep_ret, ep_len))
            i_episode += 1
            tot_ep_ret += ep_ret
            o = env_.reset()
            ep_ret = 0
            ep_len = 0
