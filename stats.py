# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np
import tqdm

import dataloader

# set skip_norm as True only when you are computing the normalization stats

audio_conf = {'target_length': 992, 'melbins':128, 'skip_norm': True, 'mode': 'train', 'dataset': 'audioset', 'root': 'unbal'}
# It is only for statistics and has nothing to do with the main file
# Select the directory in the main file format
train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetSpec('un_train_index_cleaned.csv', label_csv='class_labels_indices.csv',
                                audio_conf=audio_conf), batch_size=128, shuffle=False, num_workers=24, pin_memory=True)
# waveform = torch.randn(1, 164200).cuda()
import torchaudio
# fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
#                                                   window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
# print(fbank.shape)

pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
mean=[]
std=[]
for i, (audio_input, labels) in pbar:
    # audio_input.cuda()
    # audio_input = torch.squeeze(audio_input, dim=1)
    # # print(audio_input.shape)
    # audio_input = torchaudio.compliance.kaldi.fbank(audio_input, htk_compat=True, sample_frequency=16000, use_energy=False,
    #                                                 window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    pbar.set_postfix({'mean': f'{cur_mean:.6f}', 'std': f'{cur_std:.6e}'})
print(np.mean(mean), np.mean(std))
print("mean std:", np.std(mean))
print("std  std:", np.std(std))
np.savez('stats_16k.npz', mean=mean, std=std)
# -3.1143048 3.948369 AS2M 2D
# -9.419433e-12 0.149615 AS20K 1D
# -0.00022921551 0.14982459 AS20K 1D w/o sample mean
# -0.00026077428 0.15974662 AS2M 1D w/o sample mean
# -4.256705 4.510559
# mean std: 0.24865827
# std  std: 0.13672103 # AS2M 2D w/o norm, 32000Hz

# -5.9041724 3.4507358
# mean std: 2.900047
# std  std: 1.0630941 # AS2M 2D w/o norm, 16000Hz

# -4.6088505 4.533614
# mean std: 0.25787365
# std  std: 0.13344985 AS20K 16KHZ

# -4.307513 4.325253
# mean std: 0.24494013
# std  std: 0.13997129 AS2M 1024 

# -4.421761 4.32408
# mean std: 0.25143346
# std  std: 0.14057222
