import torch
import numpy as np
import yaml
import os


from model import MLP
import torch.nn
# from utils import sample_batch, get_env_dim, load_demo, sample_demo, check_path, separate_goal, tensor_from_numpy
from utils import *
PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
YAML_PATH = PROJECT_PATH + "/test.yaml"
DEMO_PATH = PROJECT_PATH + "/demo/"
RESULT_PATH = PROJECT_PATH + "/result/"

def mse_loss(pred, target):
    loss_func = torch.nn.MSELoss()
    loss = loss_func(pred, target)
    return loss

if __name__ == "__main__":
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    exp_demo_file_name_ = args['exp_demo_file_name']
    lea_demo_file_name_ = args['lea_demo_file_name']

    exp_policy_file_ = POLICY_PATH + exp_policy_file_name_ + ".pt"
    exp_demo_file_ = DEMO_PATH + exp_demo_file_name_ + ".pkl"
    lea_demo_file_ = DEMO_PATH + lea_demo_file_name_ + ".pkl"
    
    P_hidden_layers_ = args['P_hidden_layers']
    f_hidden_layers_ = args['f_hidden_layers']
    g_hidden_layers_ = args['g_hidden_layers']
    D_hidden_layers_ = args['D_hidden_layers']

    P_options_ = args['P_options']
    f_options_ = args['f_options']
    g_options_ = args['g_options']
    D_options_ = args['D_options']

    learning_rate_ = args['learning_rate']
    epochs_ = args['epochs']
    batch_size_ = args['batch_size']

    lambda_1_ = args['lambda_1']



    # initialize environment

    # load saved policy
    pi = torch.load(exp_policy_file_).to(device=device_)
    P = MLP(lea_obs_dim_ + lea_act_dim_ - lea_goal_dim_, lea_obs_dim_ - lea_goal_dim_, P_hidden_layers_, learning_rate_, device_, P_options_).to(device=device_)
    f = MLP(lea_obs_dim_ - lea_goal_dim_, exp_obs_dim_ - exp_goal_dim_, f_hidden_layers_, learning_rate_, device_, f_options_).to(device=device_)
    g = MLP(exp_act_dim_, lea_act_dim_, g_hidden_layers_, learning_rate_, device_, g_options_).to(device=device_)
    D = MLP(2 * exp_obs_dim_ + exp_act_dim_ - 2 * exp_goal_dim_, 1, D_hidden_layers_, learning_rate_, device_, D_options_).to(device=device_)

    exp_demo_ = load_demo(exp_demo_file_)
    lea_demo_ = load_demo(lea_demo_file_)
    # train dynamics model
    for i_epoch in range(epochs_):
        P.optimizer.zero_grad()
        # select batch from lea domain
        sy, ay, nsy = sample_demo(lea_demo_, batch_size_)

        if lea_goal_:
            sy_goal, sy = separate_goal(sy, lea_goal_offset_, lea_goal_dim_)
            nsy_goal, nsy = separate_goal(nsy, lea_goal_offset_, lea_goal_dim_)
        sy_goal = tensor_from_numpy(sy_goal, device_)
        sy = tensor_from_numpy(sy, device_)
        ay = tensor_from_numpy(ay, device_)
        nsy_goal = tensor_from_numpy(nsy_goal, device_)
        nsy = tensor_from_numpy(nsy, device_)

        
        
        p_input = torch.cat((sy, ay), 1)
        nsy_hat = P(p_input)
        print(nsy.shape)
        print(nsy_hat.shape)
        loss = mse_loss(nsy_hat, nsy)
        loss.backward()
        P.optimizer.step()
        print(loss.item())

    
    for i_epoch in range(epochs_):
        P.optimizer.zero_grad()
        f.optimizer.zero_grad()
        g.optimizer.zero_grad()
        D.optimizer.zero_grad()

        sx, ax, nsx = sample_demo(exp_demo_, batch_size_)
        sy, ay, nsy = sample_demo(lea_demo_, batch_size_)
        
        if exp_goal_:
            sx_goal, sx = separate_goal(sx, exp_goal_offset_, exp_goal_dim_)
            nsx_goal, nsx = separate_goal(nsx, exp_goal_offset_, exp_goal_dim_)
        if lea_goal_:
            sy_goal, sy = separate_goal(sy, lea_goal_offset_, lea_goal_dim_)
            nsy_goal, nsy = separate_goal(nsy, lea_goal_offset_, lea_goal_dim_)

        sx_goal = tensor_from_numpy(sx_goal, device_)
        sx = tensor_from_numpy(sx, device_)
        ax = tensor_from_numpy(ax, device_)
        nsx_goal = tensor_from_numpy(nsx_goal, device_)
        nsx = tensor_from_numpy(nsx, device_)

        sy_goal = tensor_from_numpy(sy_goal, device_)
        sy = tensor_from_numpy(sy, device_)
        ay = tensor_from_numpy(ay, device_)
        nsy_goal = tensor_from_numpy(nsy_goal, device_)
        nsy = tensor_from_numpy(nsy, device_)


        d_input = torch.cat((sx, ax, nsx), 1)
        result1 = D(d_input)
        loss1 = torch.mean(torch.log(result1))

        sx_hat = f(sy)
        sx_hat_goal = torch.cat((sx_hat[:,:exp_goal_offset_], sy_goal, sx_hat[:,exp_goal_offset_:]), 1)
        ax_hat = pi.act(sx_hat_goal)
        ay_hat = g(ax_hat)
        nsx_hat = f(P(torch.cat((sy, g(ax_hat)), 1)))
        
        d_input = torch.cat((sx_hat, ax_hat, nsx_hat), 1)
        result2 = D(d_input)

        loss2 = torch.mean(torch.log(1-result2))
        loss = loss1 + loss2

        loss.backward()
        D.optimizer.step()

        P.optimizer.zero_grad()
        f.optimizer.zero_grad()
        g.optimizer.zero_grad()
        D.optimizer.zero_grad()

        sx_hat = f(sy)
        sx_hat_goal = torch.cat((sx_hat[:,:exp_goal_offset_], sy_goal, sx_hat[:,exp_goal_offset_:]), 1)
        ax_hat = pi.act(sx_hat_goal)
        ay_hat = g(ax_hat)
        nsx_hat = f(P(torch.cat((sy, g(ax_hat)), 1)))
        
        d_input = torch.cat((sx_hat, ax_hat, nsx_hat), 1)
        result3 = D(d_input)

        loss3 = torch.mean(torch.log(result3))
        loss4 = mse_loss(ay_hat, ay)
        loss = loss3 + lambda_1_ * loss4

        print("loss1: %.3f, loss2: %.3f, loss3: %.3f, loss4: %.3f" %(loss1.item(), loss2.item(), loss3.item(), loss4.item()))

        loss.backward()
        # print(f.fc[0].weight.grad)
        # for n,p in f.named_parameters():
        #     print(p.grad.abs().mean())
        f.optimizer.step()
        g.optimizer.step()
        print(i_epoch)
        

    torch.save(P.state_dict(), result_path_ + "P.pt")
    torch.save(f.state_dict(), result_path_ + "f.pt")
    torch.save(g.state_dict(), result_path_ + "g.pt")
    torch.save(D.state_dict(), result_path_ + "D.pt")
