
import os
import torch
import numpy as np
from models import TrajectoryGenerator
from config import conf_test
from torch.utils.data import DataLoader,random_split
from trajectories_data import TrajectoryDataset

class Evaluate:
    def seq_collate(self,data):
        (obs_seq_list, pred_seq_list, loss_mask_list) = zip(*data)
        #obs_seq_list shape (batch, banyak orang, banyak fitur, panjang sequence)
        _len = [len(seq) for seq in obs_seq_list]
        cum_start_idx = [0] + np.cumsum(_len).tolist()

        #nandain mana orang yang bersama-sama ada di dalam suatu scene
        seq_start_end = [[start, end]
                        for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        # Data format: batch, features, seq_len
        # LSTM input format: seq_len, batch, features

        # diubah jadi (batch * banyak orang per scene, banyak fitur, panjang sequence )
        # dan diubah urutannya pke permute
        # jadi (panjang sequence, batch * banyak orang per scene, banyak fitur)
        obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
        pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
        loss_mask = torch.cat(loss_mask_list, dim=0)
        seq_start_end = torch.LongTensor(seq_start_end)
        out = [
            obs_traj, pred_traj, loss_mask, seq_start_end
        ]

        return tuple(out)

    def data_loader(self,path, conf_train):
        dset = TrajectoryDataset(
            path,
            obs_len=conf_train['OBS_LEN'],
            pred_len=conf_train['PRED_LEN'],
            skip=conf_train['SKIP'],
            delim=conf_train['DELIM'])
        
        if 'gveii' in path:
            generator1 = torch.Generator().manual_seed(42)
            train,test,val,_ = random_split(dset,[0.5,0.16,0.16,0.18],generator=generator1)
            loader_test = DataLoader(
                test,
                batch_size=conf_train['BATCH_SIZE'],
                shuffle=True,
                pin_memory=True,
                num_workers=conf_train['LOADER_NUM_WORKERS'],
                collate_fn=self.seq_collate)
            return loader_test
        else:
            loader_test = DataLoader(
                dset,
                batch_size=conf_train['BATCH_SIZE'],
                shuffle=True,
                num_workers=conf_train['LOADER_NUM_WORKERS'],
                collate_fn=self.seq_collate)
            return loader_test
    
    def displacement_error(self,pred_traj, pred_traj_gt, mode='sum'):
        """
        Input:
        - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
        - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
        predictions.
        - mode: Can be one of sum, raw
        Output:
        - loss: gives the euclidian displacement error
        """
        loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
        loss = loss**2
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
        if mode == 'sum':
            return torch.sum(loss)
        elif mode == 'raw':
            # 1d tensor isinya error per orang
            return loss
        
    def final_displacement_error(self,pred_pos, pred_pos_gt, mode='sum'):
        """
        Input:
        - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
        - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
        last pos
        - consider_ped: Tensor of shape (batch)
        Output:
        - loss: gives the euclidian displacement error
        """
        loss = pred_pos_gt - pred_pos
        loss = loss**2
        loss = torch.sqrt(loss.sum(dim=1))
        if mode == 'raw':
            return loss
        else:
            return torch.sum(loss)
    
    def get_generator(self,checkpoint):
        conf = checkpoint['args']
        generator = TrajectoryGenerator(
            obs_len=conf['OBS_LEN'],
            pred_len=conf['PRED_LEN'],
            embedding_dim=conf['EMBEDDING_DIM'],
            encoder_h_dim=conf['ENCODER_H_DIM_G'],
            noise_type=conf['NOISE_TYPE'],
            decoder_h_dim=conf['DECODER_H_DIM_G'],
            mlp_dim=conf['MLP_DIM'],
            noise_dim=conf['NOISE_DIM'],
            dropout=0,
            bottleneck_dim=conf['BOTTLENECK_DIM'],
            pool_every_time_step=False,
            weight_pooling_features=conf['WEIGHT_POOLING_FEATURES'],
            degree_of_vision=conf['DEGREE_OF_VISION'],
            activation=conf['ACTIVATION'],
            batch_norm=False,
            model=conf['MODEL'],
            num_layers=1
        )

        generator.load_state_dict(checkpoint['g_state'])
        generator.cuda()
        generator.eval()
        return generator

    def evaluate_helper(self,error, seq_start_end, option='min'):
        if(option == 'min'):
            sum_ = 0
            error = torch.stack(error, dim=1)
            idx_min_scene = []

            for (start, end) in seq_start_end:
                start = start.item()
                end = end.item()
                _error = error[start:end]
                _error = torch.sum(_error, dim=0)
                _error = torch.min(_error)
                idx_min = torch.argmin(_error)
                idx_min_scene.append(idx_min)
                sum_ += _error
            return sum_, idx_min_scene
        elif(option == 'rank'):
            error = torch.stack(error, dim=1)
            sum_error = 0
            idx_min_scene = []
            """
            di ranking minimum setiap orang, ada di pengambilan ke berapa,
            lalu di cari yang kurang lebih setiap orangnya minimum dalam sebuah scene.
            Nilai minimum itulah yang diambil
            """
            for(start,end) in seq_start_end:
                start = start.item()
                end = end.item()
                _error = error[start:end]
                
                #cari di index ke berapa yang rata-rata dari tiap orang minimum
                rank_error = torch.argsort(_error)
                rank_error+=1
                rank_error = torch.sum(rank_error,dim=0)
                idx_min = torch.min(rank_error,dim=0).indices
                idx_min_scene.append(idx_min)
                
                #ambil error sesuai index
                _error = torch.sum(_error, dim=0)
                sum_error+=_error[idx_min]
            
            return sum_error,idx_min_scene

    def evaluate(self,conf_train, loader, generator, conf_test):
        num_samples = conf_test['NUM_SAMPLES']
        ade_outer = []
        fde_outer = []

        total_traj = 0
        idx_scene = 0
        with torch.no_grad():
            for batch in loader:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, loss_mask, seq_start_end) = batch

                ade = []
                fde = []
                pred_traj_fakes = []

                total_traj += pred_traj_gt.size(1)
                for _ in range(num_samples):
                    pred_traj_fake = generator(obs_traj, seq_start_end)
                    pred_traj_fakes.append(pred_traj_fake)
                    ade.append(self.displacement_error(
                        pred_traj=pred_traj_fake,
                        pred_traj_gt=pred_traj_gt[:,:,:2],
                        mode='raw'
                    ))
                    fde.append(self.final_displacement_error(
                        pred_pos=pred_traj_fake[-1],
                        pred_pos_gt=pred_traj_gt[-1,:,:2],
                        mode='raw'
                    ))
                eval_help_ade = self.evaluate_helper(error=ade,seq_start_end=seq_start_end, option=conf_test['EVAL_TYPE'])
                ade_sum = eval_help_ade[0]
                eval_help_fde = self.evaluate_helper(error=fde,seq_start_end=seq_start_end, option=conf_test['EVAL_TYPE'])
                fde_sum = eval_help_fde[0]
                
                ade_min_idx = eval_help_ade[1]
                
                #for plotting
                idx_plot = 0
                for (start,end) in seq_start_end:
                    directory = 'plot_{}/scene_{}'.format(conf_test['MODEL_NAME'],idx_scene)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        
                    file_traj_gt_path = os.path.join(directory,'{}_traj_gt.npy'.format(conf_test['MODEL_NAME']))
                    file_pred_fake_path = os.path.join(directory,'{}_pred_fake.npy'.format(conf_test['MODEL_NAME']))
                    
                    min_best_k = ade_min_idx[idx_plot].item()
                    pred_traj_gt_plot = pred_traj_gt[:,start:end,:2]
                    pred_traj_fake_plot = pred_traj_fakes[min_best_k]
                    pred_traj_fake_plot = pred_traj_fake_plot[:,start:end,:]
                    obs_traj_plot = obs_traj[:,start:end,:2]
                    traj_real_plot = torch.cat([obs_traj_plot,pred_traj_gt_plot],dim=0)
                    pred_traj_fake_plot = torch.cat([obs_traj_plot,pred_traj_fake_plot],dim=0)
                    
                    # shape (batch, feature, seq_len)
                    traj_real_plot = traj_real_plot.permute(1,2,0)
                    pred_traj_fake_plot = pred_traj_fake_plot.permute(1,2,0)
                    
                    traj_real_plot = traj_real_plot.cpu().numpy()
                    pred_traj_fake_plot = pred_traj_fake_plot.cpu().numpy()
                    
                    np.save(file_traj_gt_path,traj_real_plot)
                    np.save(file_pred_fake_path,pred_traj_fake_plot)
                    
                    idx_scene+=1
                    idx_plot+=1

                ade_outer.append(ade_sum)
                fde_outer.append(fde_sum)
            ade = sum(ade_outer)/(total_traj*conf_train['PRED_LEN'])
            fde = sum(fde_outer)/total_traj
            return ade,fde
        
    def main(self, conf_test):
        if os.path.isdir(conf_test['MODEL_PATH']):
            filenames = os.listdir(conf_test['MODEL_PATH'])
            filenames.sort()
            paths = [
                os.path.join(conf_test['MODEL_PATH'],file_)for file_ in filenames
            ]
        else:
            paths = [conf_test['MODEL_PATH']]

        for path in paths:
            checkpoint = torch.load(path)
            generator = self.get_generator(checkpoint=checkpoint)
            conf_train = checkpoint['args']
            path = conf_test['TEST_DSET_PATH']
            loader = self.data_loader(path,conf_train=conf_train)
            ade, fde = self.evaluate(conf_train=conf_train,
                                loader=loader,
                                generator=generator,
                                conf_test=conf_test
                                )
            print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
                conf_test['DSET_NAME'], conf_train['PRED_LEN'], ade, fde))
        
if __name__ == '__main__':
    evaluate = Evaluate()
    evaluate.main(conf_test)
