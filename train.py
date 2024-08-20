from collections import defaultdict
from copy import deepcopy
import gc
import logging
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from config import conf_train
import numpy as np
from torch.utils.data import DataLoader,random_split
from trajectories_data import TrajectoryDataset

from models import TrajectoryGenerator, TrajectoryDiscriminator

class Train:

    torch.backends.cudnn.benchmark = True

    FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    
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
        obs_traj.pin_memory()
        pred_traj.pin_memory()
        loss_mask.pin_memory()
        seq_start_end.pin_memory()
        out = [
            obs_traj, pred_traj, loss_mask, seq_start_end
        ]

        return tuple(out)

    def data_loader(self, conf_train):
        if 'gveii' in conf_train['TRAIN_DSET_PATH']:
            
            dset = TrajectoryDataset(
            conf_train['TRAIN_DSET_PATH'],
            obs_len=conf_train['OBS_LEN'],
            pred_len=conf_train['PRED_LEN'],
            skip=conf_train['SKIP'],
            delim=conf_train['DELIM'])
            generator1 = torch.Generator().manual_seed(42)
            
            train,test,val,_ = random_split(dset,[0.5,0.16,0.16,0.18],generator=generator1)

            loader_train = DataLoader(
                train,
                batch_size=conf_train['BATCH_SIZE'],
                shuffle=True,
                pin_memory=True,
                num_workers=conf_train['LOADER_NUM_WORKERS'],
                collate_fn=self.seq_collate)
            
            loader_val = DataLoader(
                val,
                batch_size=conf_train['BATCH_SIZE'],
                shuffle=True,
                pin_memory=True,
                num_workers=conf_train['LOADER_NUM_WORKERS'],
                collate_fn=self.seq_collate)
            return loader_train,loader_val
        else:
            dset_train = TrajectoryDataset(
                conf_train['TRAIN_DSET_PATH'],
                obs_len=conf_train['OBS_LEN'],
                pred_len=conf_train['PRED_LEN'],
                skip=conf_train['SKIP'],
                delim=conf_train['DELIM'])
            
            loader_train = DataLoader(
                dset_train,
                batch_size=conf_train['BATCH_SIZE'],
                shuffle=True,
                pin_memory=True,
                num_workers=conf_train['LOADER_NUM_WORKERS'],
                collate_fn=self.seq_collate)
            
            dset_val = TrajectoryDataset(
                conf_train['VAL_DSET_PATH'],
                obs_len=conf_train['OBS_LEN'],
                pred_len=conf_train['PRED_LEN'],
                skip=conf_train['SKIP'],
                delim=conf_train['DELIM'])
            
            loader_val = DataLoader(
                dset_val,
                batch_size=conf_train['BATCH_SIZE'],
                shuffle=True,
                pin_memory=True,
                num_workers=conf_train['LOADER_NUM_WORKERS'],
                collate_fn=self.seq_collate)
            return loader_train, loader_val
            

    def init_weights(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight)
            
    def get_dtypes(self, conf_train):
        long_dtype = torch.LongTensor
        float_dtype = torch.FloatTensor
        if(conf_train['USE_GPU'] == 1):
            long_dtype = torch.cuda.LongTensor
            float_dtype = torch.cuda.FloatTensor
        return long_dtype, float_dtype

    def main(self, conf_train):
        
        long_dtype, float_dtype = self.get_dtypes(conf_train)

        device = torch.device("cuda")
        
        print("Initializing train and val dataset")
        
        train_loader,val_loader = self.data_loader(conf_train)
        
        generator = TrajectoryGenerator(
            obs_len=conf_train['OBS_LEN'],
            pred_len=conf_train['PRED_LEN'],
            embedding_dim=conf_train['EMBEDDING_DIM'],
            encoder_h_dim=conf_train['ENCODER_H_DIM_G'],
            noise_type=conf_train['NOISE_TYPE'],
            decoder_h_dim=conf_train['DECODER_H_DIM_G'],
            mlp_dim=conf_train['MLP_DIM'],
            noise_dim=conf_train['NOISE_DIM'],
            dropout=0,
            bottleneck_dim=conf_train['BOTTLENECK_DIM'],
            pool_every_time_step=False,
            weight_pooling_features=conf_train['WEIGHT_POOLING_FEATURES'],
            degree_of_vision=conf_train['DEGREE_OF_VISION'],
            activation=conf_train['ACTIVATION'],
            model=conf_train['MODEL'],
            batch_norm=False,
            num_layers=1
        )
        generator.to(device=device)
        
        generator.apply(self.init_weights)
        generator.type(float_dtype).train()
        print('Here is the generator:')
        print(generator)
        
        discriminator =  TrajectoryDiscriminator(
            obs_len=conf_train['OBS_LEN'],
            pred_len=conf_train['PRED_LEN'],
            embedding_dim=conf_train['EMBEDDING_DIM'],
            h_dim=conf_train['ENCODER_H_DIM_D'],
            mlp_dim=conf_train['MLP_DIM'],
            num_layers=1,
            activation=conf_train['ACTIVATION'],
            batch_norm=False,
            dropout=0
        )
        discriminator.to(device=device)
        discriminator.apply(self.init_weights)
        discriminator.type(float_dtype).train()
        print('Here is the discriminator:')
        print(discriminator)
        
        g_loss_fn = self.gan_g_loss
        d_loss_fn = self.gan_d_loss
        
        optimizer_g = optim.Adam(generator.parameters(), lr=conf_train['G_LEARNING_RATE'])
        optimizer_d = optim.Adam(discriminator.parameters(), lr=conf_train['D_LEARNING_RATE'])
        
        t,epoch = 0,0

        restore_path = None
        if conf_train['PATH_CHECKPOINT'] is not None:
            restore_path = conf_train['PATH_CHECKPOINT']
        elif conf_train['RESTORE_FROM_CHECKPOINT'] == True:
            restore_path = os.path.join(conf_train['OUTPUT_DIR'], 
                                        '%s_with_model.pt' % conf_train['CHECKPOINT_NAME'])
            
        if restore_path is not None and os.path.isfile(restore_path):
            print('Restoring from checkpoint {}'.format(restore_path))
            checkpoint = torch.load(restore_path)
            generator.load_state_dict(checkpoint['g_state'])
            discriminator.load_state_dict(checkpoint['d_state'])
            optimizer_g.load_state_dict(checkpoint['g_optim_state'])
            optimizer_d.load_state_dict(checkpoint['d_optim_state'])
            t = checkpoint['counters']['t']
            epoch = checkpoint['counters']['epoch']
            checkpoint['restore_ts'].append(t)
        else:
            # Starting from scratch, so initialize checkpoint data structure
            t, epoch = 0, 0
            checkpoint = {
                'args': conf_train,
                'G_losses': defaultdict(list),
                'D_losses': defaultdict(list),
                'losses_ts': [],
                'metrics_val': defaultdict(list),
                'metrics_train': defaultdict(list),
                'sample_ts': [],
                'restore_ts': [],
                'norm_g': [],
                'norm_d': [],
                'counters': {
                    't': None,
                    'epoch': None,
                },
                'g_state': None,
                'g_optim_state': None,
                'd_state': None,
                'd_optim_state': None,
                'g_best_state': None,
                'd_best_state': None,
                'best_t': None,
            }

        while epoch < conf_train['NUM_EPOCH']:
            gc.collect()
            d_steps_left = conf_train['D_STEPS']
            g_steps_left = conf_train['G_STEPS']
            epoch+=1
            print('Starting epoch {}'.format(epoch))
            for batch in train_loader:
                if d_steps_left > 0:
                    losses_d = self.discriminator_step(batch=batch,
                                                generator=generator,
                                                discriminator=discriminator,
                                                d_loss_fn=d_loss_fn,
                                                optimizer_d=optimizer_d,
                                                conf_train=conf_train
                                                )
                    d_steps_left -=1
                elif g_steps_left > 0:
                    losses_g = self.generator_step(batch=batch,
                                            generator=generator,
                                            discriminator=discriminator,
                                            g_loss_fn=g_loss_fn,
                                            optimizer_g=optimizer_g,
                                            conf_train=conf_train
                                            )
                    g_steps_left -=1
                    
                if d_steps_left > 0 or g_steps_left > 0:
                    continue

                # Maybe save loss
                if t % conf_train['PRINT_EVERY'] == 0:
                    print('t = {}'.format(t + 1))
                    for k, v in sorted(losses_d.items()):
                        print('  [D] {}: {:.3f}'.format(k, v))
                        checkpoint['D_losses'][k].append(v)
                    for k, v in sorted(losses_g.items()):
                        print('  [G] {}: {:.3f}'.format(k, v))
                        checkpoint['G_losses'][k].append(v)
                    checkpoint['losses_ts'].append(t)
                
                t += 1
                d_steps_left = conf_train['D_STEPS']
                g_steps_left = conf_train['G_STEPS']
                
            # Maybe save a checkpoint
            if epoch > 0 and epoch % conf_train['CHECKPOINT_EVERY'] == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                print('Checking stats on val ...')
                metrics_val = self.check_accuracy(
                    val_loader, generator, discriminator, d_loss_fn, conf_train
                )
                print('Checking stats on train ...')
                metrics_train = self.check_accuracy(
                    train_loader, generator, discriminator,
                    d_loss_fn, conf_train, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    print('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])

                if metrics_val['ade'] == min_ade:
                    print('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = deepcopy(generator.state_dict())
                    checkpoint['d_best_state'] = deepcopy(discriminator.state_dict())

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    conf_train['OUTPUT_DIR'], '%s_with_model.pt' % conf_train['CHECKPOINT_NAME']
                )
                print('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                print('Done.')
                    
    def discriminator_step(self,batch, generator, discriminator, d_loss_fn, optimizer_d, conf_train):
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, loss_mask, seq_start_end) = batch
        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)
        generator_out = generator(obs_traj, seq_start_end)
        pred_traj_fake = generator_out
        traj_real = torch.cat([obs_traj[:,:,:2],pred_traj_gt[:,:,:2]], dim=0)
        traj_fake = torch.cat([obs_traj[:,:,:2], pred_traj_fake[:,:,:2]], dim=0)
        
        scores_fake = discriminator(traj_fake)
        scores_real = discriminator(traj_real)
        
        data_loss = d_loss_fn(scores_real, scores_fake)
        losses['D_data_loss'] = data_loss.item()
        loss+=data_loss
        losses['D_total_loss'] = loss.item()
        optimizer_d.zero_grad()
        loss.backward()
        
        if conf_train['CLIPPING_THRESHOLD_D'] > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(),conf_train['CLIPPING_THRESHOLD_D'])
        optimizer_d.step()

        return losses

    def generator_step(self,batch, generator, discriminator, g_loss_fn, optimizer_g, conf_train):
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, loss_mask, seq_start_end) = batch
        pred_traj_gt = pred_traj_gt[:,:,:2]
        losses={}
        loss = torch.zeros(1).to(pred_traj_gt)
        
        loss_mask = loss_mask[:, conf_train['OBS_LEN']:]
        
        g_l2_loss = []
        
        for _ in range(conf_train['BESK_K']):
            generator_out = generator(obs_traj, seq_start_end)
            
            pred_traj_fake = generator_out
            
            if conf_train['L2_LOSS_WEIGHT'] > 0:
                g_l2_loss.append(conf_train['L2_LOSS_WEIGHT'] * self.l2_loss(
                    pred_traj=pred_traj_fake,
                    pred_traj_gt=pred_traj_gt,
                    loss_mask=loss_mask,
                    mode='raw'
                ))
                
        g_l2_loss_sum = torch.zeros(1).to(pred_traj_gt)
        if conf_train['L2_LOSS_WEIGHT'] > 0:
            g_l2_loss = torch.stack(g_l2_loss, dim=1)
            for start,end in seq_start_end.data:
                _g_l2_loss = g_l2_loss[start:end]
                _g_l2_loss = torch.sum(_g_l2_loss, dim=0)
                _g_l2_loss = torch.min(_g_l2_loss) / torch.sum(
                    loss_mask[start:end])
                g_l2_loss_sum += _g_l2_loss
            losses['G_l2_loss'] = g_l2_loss_sum.item()
            loss += g_l2_loss_sum
        
        traj_fake = torch.cat([obs_traj[:,:,:2], pred_traj_fake], dim=0)
        scores_fake = discriminator(traj_fake)
        discriminator_loss = g_loss_fn(scores_fake)
        loss += discriminator_loss
        losses['G_discriminator_loss'] = discriminator_loss.item()
        losses['G_total_loss'] = loss.item()
        optimizer_g.zero_grad()
        loss.backward()
        if conf_train['CLIPPING_THRESHOLD_G'] > 0:
            nn.utils.clip_grad_norm_(
                generator.parameters(), conf_train['CLIPPING_THRESHOLD_G']
            )
        optimizer_g.step()

        return losses

    def cal_l2_losses(
            self, pred_traj_gt, pred_traj_fake, loss_mask
    ):
        g_l2_loss = self.l2_loss(
            pred_traj=pred_traj_fake, pred_traj_gt=pred_traj_gt, 
            loss_mask=loss_mask, mode='sum'
        )

        return g_l2_loss

    def cal_ade(self,pred_traj_gt, pred_traj_fake):
        ade = self.displacement_error(pred_traj=pred_traj_fake, pred_traj_gt=pred_traj_gt)
        return ade

    def cal_fde(self,pred_traj_gt, pred_traj_fake):
        fde = self.final_displacement_error(pred_pos=pred_traj_fake[-1], pred_pos_gt=pred_traj_gt[-1])
        return fde

    def check_accuracy(
        self,loader, generator, discriminator, d_loss_fn, conf_train,limit=False      
    ):
        d_losses = []
        metrics = {}
        g_l2_losses = []
        disp_error = []
        f_disp_error = []
        total_traj = 0
        loss_mask_sum = 0

        generator.eval()

        with torch.no_grad():
            for batch in loader:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, loss_mask, seq_start_end) = batch
                loss_mask = loss_mask[:, conf_train['OBS_LEN']:]

                pred_traj_fake = generator(obs_traj, seq_start_end)
                g_l2_loss = self.cal_l2_losses(pred_traj_fake=pred_traj_fake, 
                                        pred_traj_gt=pred_traj_gt[:,:,:2],
                                        loss_mask=loss_mask)
                
                ade = self.cal_ade(pred_traj_fake=pred_traj_fake, pred_traj_gt=pred_traj_gt[:,:,:2])

                fde = self.cal_fde(pred_traj_fake=pred_traj_fake, pred_traj_gt=pred_traj_gt[:,:,:2])

                traj_real = torch.cat([obs_traj[:,:,:2], pred_traj_gt[:,:,:2]], dim=0)
                traj_fake = torch.cat([obs_traj[:,:,:2], pred_traj_fake], dim=0)

                scores_fake = discriminator(traj_fake)
                scores_real = discriminator(traj_real)

                d_loss = d_loss_fn(scores_real,scores_fake)

                d_losses.append(d_loss.item())
                g_l2_losses.append(g_l2_loss.item())
                disp_error.append(ade.item())
                f_disp_error.append(fde.item())

                loss_mask_sum += torch.numel(loss_mask.data)
                total_traj += pred_traj_gt.size(1)

                if limit and total_traj >= conf_train['NUM_SAMPLE_CHECK']:
                    break
        metrics['d_loss'] = sum(d_losses) / len(d_losses)
        metrics['g_l2_loss'] = sum(g_l2_losses) / loss_mask_sum

        metrics['ade'] = sum(disp_error) / (total_traj * conf_train['PRED_LEN'])
        metrics['fde'] = sum(f_disp_error) / total_traj
        generator.train()
        return metrics
    
    def bce_loss(self,input, target):
        """
        Numerically stable version of the binary cross-entropy loss function.
        As per https://github.com/pytorch/pytorch/issues/751
        See the TensorFlow docs for a derivation of this formula:
        https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        Input:
        - input: PyTorch Tensor of shape (N, ) giving scores.
        - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

        Output:
        - A PyTorch Tensor containing the mean BCE loss over the minibatch of
        input data.
        """
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()

    def gan_g_loss(self,scores_fake):
        """
        input:
        scores_fake -> tensor (N,) containing score for fake samples
        
        output:
        loss -> tensor (,) giving gan generator loss
        """
        y_fake = torch.ones_like(scores_fake) * random.uniform(0.7,1.2)
        return self.bce_loss(scores_fake,y_fake)

    def gan_d_loss(self,scores_real, scores_fake):
        """
        input:
        scores_real -> tensor (N,) containing score for real samples
        scores_fake -> tensor (N,) containing score for fake samples
        
        output:
        loss -> tensor (,) giving gan discriminator loss
        """
        y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
        y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
        loss_real = self.bce_loss(scores_real,y_real)
        loss_fake = self.bce_loss(scores_fake,y_fake)
        return loss_real+loss_fake

    def l2_loss(self,pred_traj, pred_traj_gt, loss_mask, mode='average'):
        """
        Input:
        - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
        - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
        predictions.
        - loss_mask: Tensor of shape (batch, seq_len)
        - mode: Can be one of sum, average, raw
        Output:
        - loss: l2 loss depending on mode
        """
        loss = (loss_mask.unsqueeze(dim=2) *
                (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
        if mode == 'sum':
            return torch.sum(loss)
        elif mode == 'average':
            return torch.sum(loss) / torch.numel(loss_mask.data)
        elif mode == 'raw':
            return loss.sum(dim=2).sum(dim=1)
        
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


if __name__ == '__main__':
    train = Train()
    train.main(conf_train)