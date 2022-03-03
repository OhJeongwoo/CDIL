from copy import deepcopy
import torch
import numpy as np
import yaml
import os
import sys
import time
import matplotlib.pyplot as plt


from model import MLP, DDPGCore
import torch.nn
from utils import *
from replay_buffer import ReplayBuffer

sys.path.append(os.getcwd() + '/..')
import envs

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/test.yaml"
DEMO_PATH = PROJECT_PATH + "/demo/"
RESULT_PATH = PROJECT_PATH + "/result/"

if __name__ == "__main__":
    init_time_ = time.time()
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device_)
    
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
    exp_epi_len_ = args['exp_epi_len']

    lea_env_name_ = args['lea_env_name']
    lea_epi_len_ = args['exp_epi_len']
    lea_obs_dim_, lea_act_dim_ = get_env_dim(lea_env_name_)
    lea_goal_ = args['lea_goal']
    lea_goal_offset_ = args['lea_goal_offset']
    lea_goal_dim_ = args['lea_goal_dim']
    if exp_goal_dim_ != lea_goal_dim_:
        print("[ERR] goal dimesion is different in two domain... can not run GAMA")
        exit()

    env_ = gym.make(exp_env_name_)
    act_limit_ = env_.action_space.high[0]

    exp_policy_file_name_ = args['exp_policy_file_name']

    exp_demo_file_name_ = args['exp_demo_file_name']
    lea_demo_file_name_ = args['lea_demo_file_name']

    exp_policy_file_ = POLICY_PATH + exp_policy_file_name_ + ".pt"
    exp_demo_file_ = DEMO_PATH + exp_demo_file_name_ + ".pkl"
    lea_demo_file_ = DEMO_PATH + lea_demo_file_name_ + ".pkl"
    
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

    lambda_1_ = args['lambda_1']
    gamma_ = args['gamma']
    alpha_ = args['alpha']
    polyak_ = args['polyak']


    # hyperparameter setting

    # initialize environment
    np.random.seed(seed_)

    ac_ = DDPGCore(exp_obs_dim_, exp_act_dim_, hidden_layers_, learning_rate_, act_limit_, device_, options_).to(device_)
    ac_tar_ = deepcopy(ac_)

    for p in ac_tar_.parameters():
        p.requires_grad = False
    
    replay_buffer_ = ReplayBuffer(exp_obs_dim_, exp_act_dim_, replay_size_, device_)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac_.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_tar = ac_tar_.q(o2, ac_tar_.pi(o2))
            backup = r + gamma_ * (1 - d) * q_pi_tar

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac_.q(o, ac_.pi(o))
        return -q_pi.mean()

    def update(data):
        # First run one gradient descent step for Q.
        ac_.q.optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        ac_.q.optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac_.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        ac_.pi.optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        ac_.pi.optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac_.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac_.parameters(), ac_tar_.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak_)
                p_targ.data.add_((1 - polyak_) * p.data)


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
                batch = replay_buffer_.sample_batch(batch_size_)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch_ == 0:
            epoch = (t+1) // steps_per_epoch_
            avg_rt = test_agent()
            ts_axis.append(t+1)
            rt_axis.append(avg_rt)
            print("[%.3f] Epoch: %d, Timesteps: %d, AvgEpReward: %.3f" %(time.time() - init_time_, epoch, t+1, avg_rt))
            plt.plot(ts_axis, rt_axis)
            plt.pause(0.001)
