import numpy as np
import os
import matplotlib.pyplot as plt
import torch

def make_plot(scene_number,model_name, obs_len):
    dir_path = 'plot_{}/scene_{}'.format(model_name,scene_number)
    traj_real_path = os.path.join(dir_path,'{}_traj_gt.npy'.format(model_name))
    traj_fake_path = os.path.join(dir_path,'{}_pred_fake.npy'.format(model_name))
    
    # shape (batch, feature, seq_len)
    traj_real = torch.tensor(np.load(traj_real_path))
    traj_fake = torch.tensor(np.load(traj_fake_path))
    
    size_traj = traj_real.size(0)
    
    for i in range(size_traj):
        plt.plot(traj_real[i][0],traj_real[i][1],color='blue',marker='o')
        plt.plot(traj_fake[i][0][obs_len-1:],traj_fake[i][1][obs_len-1:],linestyle='dashed',marker='o')
        
    plt.show()
    
#contoh pemanggilan fungsi   
# make_plot(558,'exp_6_3_c',8)
    
    