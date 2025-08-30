# -*- coding: utf-8 -*-
# @Time    : 11/17/20 3:22 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : gen_weight_file.py

# gen sample weight = sum(label_weight) for label in all labels of the audio clip, where label_weight is the reciprocal of the total sample count of that class.
# Note audioset is a multi-label dataset

import argparse
import json
import numpy as np
import sys, os, csv

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def load_csv_data(csv_file, root):
    """
    Read data from a CSV file and store it in self.data
    Assumes the CSV format is:
        video_id, start_time, end_time, label1, label2, ...
    where columns after the third are labels. Modify according to your actual format.
    """
    data_list = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for line in reader:
            if len(line) < 4:
                continue
            video_id = line[0].strip()
            # Take line[3:] as label columns
            labels_str = ','.join(line[3:]).strip().strip('"')
            # labels = [lab.strip() for lab in labels_str.split(',')]
            data_list.append({
                # 'video_id': video_id,
                'wav': os.path.join(root, 'Y' + video_id + '.wav'),
                'labels': labels_str
            })
    return data_list

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_path", type=str, default='/media/yuinst/B362-Dataset/AudioSet2M/un_train_index_cleaned.csv', help="the root path of data json file")

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path

    index_dict = make_index_dict('/media/yuinst/B362-Dataset/AudioSet2M/class_labels_indices.csv')
    label_count = np.zeros(527)

    data = load_csv_data(data_path, '/media/yuinst/B362-Dataset/AudioSet2M/unbal/')

    for sample in data:
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            label_count[label_idx] = label_count[label_idx] + 1

    # the reason not using 1 is to avoid underflow for majority classes, add small value to avoid underflow
    label_weight = 1000.0 / (label_count + 0.01)
    #label_weight = 1000.0 / (label_count + 0.00)
    sample_weight = np.zeros(len(data))

    for i, sample in enumerate(data):
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            # summing up the weight of all appeared classes in the sample, note audioset is multiple-label classification
            sample_weight[i] += label_weight[label_idx]
    np.savetxt(data_path[:-4]+'_weight.csv', sample_weight, delimiter=',')