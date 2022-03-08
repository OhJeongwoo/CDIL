from copy import deepcopy
from ssl import OP_NO_TLSv1_1
import torch
import numpy as np
import yaml
import os
import sys
import time
import matplotlib.pyplot as plt
import argparse

from model import MLP, ValueDiceCore
import torch.nn
from utils import *
from replay_buffer import ReplayBuffer

sys.path.append(os.getcwd() + '/..')
import envs

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/yaml/"
DEMO_PATH = PROJECT_PATH + "/demo/"
RESULT_PATH = PROJECT_PATH + "/result/"

if __name__ == "__main__":
    init_time_ = time.time()
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Soft Actor-Critic (SAC)')
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
    exp_demo_file_name_ = args['exp_demo_file_name']
    exp_demo_file_ = DEMO_PATH + exp_demo_file_name_ + ".pkl"


    env_ = gym.make(exp_env_name_)
    act_limit_ = env_.action_space.high[0]
    
    hidden_layers_ = args['pi_hidden_layers']
    options_ = args['pi_options']
    
    replay_size_ = args['replay_size']
    learning_rate_ = args['learning_rate']
    epochs_ = args['epochs']
    batch_size_ = args['batch_size']
    iter_size_ = args['iter_size']
    n_log_epi_ = args['n_log_epi']
    steps_per_epoch_ = args['steps_per_epoch']
    start_steps_ = args['start_steps']
    update_after_ = args['update_after']
    update_every_ = args['update_every']
    save_interval_ = args['save_interval']
    seed_ = args['seed']

    gamma_ = args['gamma']
    alpha_ = args['alpha']
    

    # hyperparameter setting

    # initialize environment
    np.random.seed(seed_)

    ac_ = ValueDiceCore(exp_obs_dim_, exp_act_dim_, hidden_layers_, learning_rate_, act_limit_, device_, options_).to(device_)
    exp_demo_ = load_demo(exp_demo_file_)
    
    replay_buffer_ = ReplayBuffer(exp_obs_dim_, exp_act_dim_, replay_size_, device_)

    def update_v(exp_batch, cur_batch):
        o = cur_batch['obs']
        a = cur_batch['act']
        o2 = cur_batch['obs2']
        a2, _ = ac_.pi(o2)

        o0 = tensor_from_numpy(exp_batch['s0'], device_)
        a0, _ = ac_.pi(o0)
        oe = tensor_from_numpy(exp_batch['obs'], device_)
        ae = tensor_from_numpy(exp_batch['act'], device_)
        oe2 = tensor_from_numpy(exp_batch['nobs'], device_)
        ae2, _ = ac_.pi(oe2)

        v = ac_.v(o, a)
        v2 = ac_.v(o2, a2)
        v0 = ac_.v(o0, a0)
        ve = ac_.v(oe, ae)
        ve2 = ac_.v(oe2, ae2)

        value_log = (1-alpha_) * torch.exp(ve - gamma_ * ve2) + alpha_ * torch.exp(v - gamma_ * v2)
        loss_log = torch.log(torch.mean(value_log))
        value_linear = (1-alpha_) * (1 - gamma_) * v0 + alpha_ * (v - gamma_ * v2)
        loss_linear = torch.mean(value_linear)
        loss = loss_log - loss_linear
        ac_.v.optimizer.zero_grad()
        loss.backward()
        ac_.v.optimizer.step()

    def update_pi(exp_batch, cur_batch):
        o = cur_batch['obs']
        a = cur_batch['act']
        o2 = cur_batch['obs2']
        a2, _ = ac_.pi(o2)

        o0 = tensor_from_numpy(exp_batch['s0'], device_)
        a0, _ = ac_.pi(o0)
        oe = tensor_from_numpy(exp_batch['obs'], device_)
        ae = tensor_from_numpy(exp_batch['act'], device_)
        oe2 = tensor_from_numpy(exp_batch['nobs'], device_)
        ae2, _ = ac_.pi(oe2)

        v = ac_.v(o, a)
        v2 = ac_.v(o2, a2)
        v0 = ac_.v(o0, a0)
        ve = ac_.v(oe, ae)
        ve2 = ac_.v(oe2, ae2)

        value_log = (1-alpha_) * torch.exp(ve - gamma_ * ve2) + alpha_ * torch.exp(v - gamma_ * v2)
        loss_log = torch.log(torch.mean(value_log))
        value_linear = (1-alpha_) * (1 - gamma_) * v0 + alpha_ * (v - gamma_ * v2)
        loss_linear = torch.mean(value_linear)
        loss = loss_log - loss_linear
        loss = -loss
        ac_.pi.optimizer.zero_grad()
        loss.backward()
        ac_.pi.optimizer.step()

        
    def get_action(o, remove_grad=True):
        a = ac_.act(torch.unsqueeze(torch.as_tensor(o, dtype=torch.float32), dim=0).to(device=device_))
        if remove_grad:
            return a.detach().cpu().numpy()
        return a

        
    def test_agent():
        tot_ep_ret = 0.0
        for _ in range(n_log_epi_):
            o, d, ep_ret, ep_len = env_.reset(), False, 0, 0
            while not(d or (ep_len == exp_epi_len_)):
                # Take deterministic actions at test time 
                o, r, d, _ = env_.step(get_action(o))
                ep_ret += r
                ep_len += 1
            tot_ep_ret += ep_ret
        return tot_ep_ret / n_log_epi_

    total_steps = steps_per_epoch_ * epochs_
    start_time = time.time()
    o, ep_ret, ep_len = env_.reset(), 0, 0

    ts_axis = []
    rt_axis = []
    max_avg_rt = -1000.0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps_:
            a = get_action(o)
        else:
            a = env_.action_space.sample()

        # Step the env
        o2, r, d, _ = env_.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == exp_epi_len_ else d

        # Store experience to replay buffer
        replay_buffer_.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == exp_epi_len_):
            o, ep_ret, ep_len = env_.reset(), 0, 0

        # Update handling
        if t >= update_after_ and t % update_every_ == 0:
            for j in range(update_every_):
                exp_batch = sample_geometric_demo(exp_demo_, batch_size_, gamma_)
                cur_batch = replay_buffer_.sample_batch(batch_size_)
                update_v(exp_batch, cur_batch)

                exp_batch = sample_geometric_demo(exp_demo_, batch_size_, gamma_)
                cur_batch = replay_buffer_.sample_batch(batch_size_)
                update_pi(exp_batch, cur_batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch_ == 0:
            epoch = (t+1) // steps_per_epoch_
            avg_rt = test_agent()
            ts_axis.append(t+1)
            rt_axis.append(avg_rt)
            if epoch % save_interval_ == 0:
                torch.save(ac_, POLICY_PATH + exp_name_ + "/ac_"+str(epoch).zfill(3)+".pt")
            if max_avg_rt < avg_rt:
                max_avg_rt = avg_rt
                torch.save(ac_, POLICY_PATH + exp_name_ + "/ac_best.pt")    
            print("[%.3f] Epoch: %d, Timesteps: %d, AvgEpReward: %.3f" %(time.time() - init_time_, epoch, t+1, avg_rt))
            plt.plot(ts_axis, rt_axis)
            plt.pause(0.001)
