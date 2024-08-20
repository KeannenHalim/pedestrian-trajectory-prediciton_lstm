import re
import numpy as np
import os
import math
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def read_file(self,_path, delim='\t'):
        # baca baris per baris, masukkan ke array
        data = []
        if delim == 'tab':
            delim = '\t'
        with open(_path, 'r') as f:
            for line in f:
                # ilangin white space di ujung ujungnya
                line = line.strip()
                # ubah white space sisanya jadi tab
                line = re.sub(r'\s+','\t',line)
                line = line.split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)
    
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, delim='\t'):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len

        # sampling mau setiap berapa timestep sekali
        self.skip = skip

        # panjang sliding window
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        # ambil semua isi dari data_dir
        all_files = os.listdir(self.data_dir)

        # buat path ke semua isi dari data_dir
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        num_peds_in_seq = []
        seq_list = []

        loss_mask_list = []

        for path in all_files:
            data = self.read_file(path, delim)
            """
            Isi dari data
            [
                [frame_number, pedestrian_id, pos_x, pos_y],
                [frame_number, pedestrian_id, pos_x, pos_y],
                ...
                [frame_number, pedestrian_id, pos_x, pos_y]
            ]
            """
            # ambil semua data pada kolom pertama tanpa ada duplikat (ambil frame number)
            frame_numbers = np.unique(data[:, 0]).tolist()

            frame_data = []

            total_frame_numbers = len(frame_numbers)

            for frame in frame_numbers:
                # ambil semua baris yang nomor frame nya sama / mengelompokkan pedestrian per frame
                frame_data.append(data[frame == data[:, 0]])

            """
            Isi frame_data
            [
                [
                    [frame_number_1, pedestrian_id_1, pos_x, pos_y],
                    [frame_number_1, pedestrian_id_2, pos_x, pos_y],
                    ...
                ],
                [
                    [frame_number_2, pedestrian_id_1, pos_x, pos_y],
                    [frame_number_2, pedestrian_id_2, pos_x, pos_y],
                    ...
                ]
            ]
            """

            for idx in range(0, total_frame_numbers, 1):

                skip_length = self.seq_len*self.skip
                if idx + skip_length >= total_frame_numbers:
                    break

                # ambil satu sliding window
                """
                Isi curr_seq_data
                [
                    [frame_number_1, pedestrian_id_1, pos_x, pos_y],
                    [frame_number_1, pedestrian_id_2, pos_x, pos_y],
                    [frame_number_2, pedestrian_id_1, pos_x, pos_y],
                    [frame_number_2, pedestrian_id_2, pos_x, pos_y],
                    [frame_number_3, pedestrian_id_1, pos_x, pos_y],
                    [frame_number_3, pedestrian_id_2, pos_x, pos_y],
                    ...
                ]
                """
                curr_seq_data = np.concatenate(frame_data[idx:idx + skip_length + 1], axis=0)
                
                # cari ada pedestrian siapa saja yang unik dalam satu sequence
                peds_id_in_curr_seq = np.unique(curr_seq_data[:, 1])

                # array 3d isinya nol (pedestrians,features, panjang_timesteps)
                curr_seq = np.zeros((len(peds_id_in_curr_seq), 4, self.seq_len))

                # array 2d isinya nol
                # ukurannya (pedestrians, panjang_timesteps)
                curr_loss_mask = np.zeros((len(peds_id_in_curr_seq), self.seq_len))

                num_peds_considered = 0

                for ped_id in peds_id_in_curr_seq:

                    # ambil sequence dari sebuah pedestrian id
                    """
                    Isi dari curr_ped_seq
                    [
                        [frame_number_1, pedestrian_id_1, pos_x, pos_y],
                        [frame_number_2, pedestrian_id_1, pos_x, pos_y],
                        [frame_number_3, pedestrian_id_1, pos_x, pos_y],
                        ...
                    ]
                    """
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id]

                    # cari awal pedestriannya ada di frame mana
                    pad_front = frame_numbers.index(curr_ped_seq[0, 0]) - idx

                    # cari akhir pedestriannya ada di frame mana
                    pad_end = frame_numbers.index(curr_ped_seq[-1, 0]) - idx + 1

                    # skip jika pedestriannya tidak ada sepanjang sequence
                    if pad_end - pad_front != skip_length+1:
                        continue
                    
                    curr_ped_seq_skip = []
                    for frame_number in range(0,curr_ped_seq.shape[0],1):
                        if(frame_number % self.skip == 0):
                            curr_ped_seq_skip.append(curr_ped_seq[frame_number])
                    
                    curr_ped_seq_skip = np.stack(curr_ped_seq_skip)
                    curr_ped_seq = curr_ped_seq_skip

                    # ambil fitur-fitur nya
                    """
                    Isi curr_ped_seq
                    [[f1t1,f1t2],[f2t1,f2t2],[f3t1,f3t2] ...]
                    """
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    #hitung kecepatan pada setiap timestep
                    curr_ped_seq_temp = []
                    curr_ped_seq_temp.append(np.concatenate(curr_ped_seq[:1, : self.seq_len]))
                    curr_ped_seq_temp.append(np.concatenate(curr_ped_seq[1:2, : self.seq_len]))
                    for feature in curr_ped_seq:
                        if'gveii' in data_dir:
                            temp = np.array([((next-now)/(0.125*self.skip)) for now,next in zip(feature,feature[1:])])
                        else:
                            temp = np.array([((next-now)/(0.4*self.skip)) for now,next in zip(feature,feature[1:])])
                        curr_ped_seq_temp.append(temp)

                    curr_ped_seq = np.stack(curr_ped_seq_temp)

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front: pad_end] = curr_ped_seq
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > 0:
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    #seq_list shape (banyak scene, banyak orang, banyak fitur, panjang sequence)
                    seq_list.append(curr_seq[:num_peds_considered])

        # ada berapa sequence
        self.num_seq = len(seq_list)
        #seq_list shape (banyak scene * banyak orang, banyak fitur, panjang sequence
        seq_list = np.concatenate(seq_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)

        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        
        # untuk membatasi pedestrian yang ada di dalam satu scene
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :], self.loss_mask[start:end, :]
        ]
        return out