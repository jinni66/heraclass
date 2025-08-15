import argparse
import json
import os
import sys

import pandas as pd


key_str_mapping = {
    'iscx-vpn-2016': 'vpn_label',
    'iscx-tor-2016': 'tor_label',
    'ustc-tfc-2016': 'ustc_label',
    'ios': 'ios_label',
    'android': 'android_label',
    'cic': 'cic_label',
}


def _get_file_list(file_dir):
    file_list = []
    for filepath, _, filenames in os.walk(file_dir):
        for filename in filenames:
            file_list.append(os.path.join(filepath, filename))
    return file_list


def _get_row_count(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return len(file.readlines()) - 1


def _get_label_list(key_str):
    with open(os.path.join('label_split/training', f'{key_str}.json')) as f:
        train_labels = json.load(f)
    with open(os.path.join('label_split/testing', f'{key_str}.json')) as f:
        test_labels = json.load(f)

    return train_labels, test_labels


def _get_min_label_count(file_list, label_list):
    min_len = sys.maxsize
    label_name = ''

    for label in label_list:
        cur_len = 0
        for file_path in file_list:
            if label in file_path:
                cur_len += _get_row_count(file_path)
        if cur_len < min_len:
            min_len = cur_len
            label_name = label
    return min_len, label_name


def _split_file(file_list, label, min_label_count, save_dir, dir_name, train_rate=0.7, val_rate=0.2, test_rate=0.1, need_train=False):
    print(f'Begin handle label {label}, min_label_count {min_label_count}')
    df_list = []
    for file_path in file_list:
        if label in file_path:
            df_list.append(pd.read_csv(file_path))
    df_raw = pd.concat(df_list)
    train_count = int(min_label_count * train_rate)
    val_count = int(min_label_count * val_rate)
    test_count = int(min_label_count * test_rate)

    train_df = df_raw[:train_count]
    val_df = df_raw[train_count:train_count + val_count]
    test_df = df_raw[train_count + val_count:train_count + val_count + test_count]

    train_save_path = os.path.join(save_dir, dir_name, label + '_train.csv')
    val_save_path = os.path.join(save_dir, dir_name, label + '_val.csv')
    test_save_path = os.path.join(save_dir, dir_name, label + '_test.csv')

    os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)

    if need_train:
        train_df.to_csv(train_save_path, index=False)
        val_df.to_csv(val_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)
    print(f'Handle label {label} success. saved in {os.path.join(save_dir, dir_name)}')


def handle_data(args, train_rate=0.7, val_rate=0.2, test_rate=0.1):
    data_dirs = args.data_dir
    for dir in data_dirs:
        file_list = _get_file_list(dir)
        dir_name = os.path.basename(os.path.dirname(file_list[0]))
        key_str = key_str_mapping[dir_name]
        train_labels, test_labels = _get_label_list(key_str)

        min_label_count, min_label_name = _get_min_label_count(file_list, test_labels)

        for label in test_labels:
            need_train = label in train_labels
            try:
                _split_file(file_list, label, min_label_count, args.save_dir, dir_name, train_rate, val_rate, test_rate, need_train)
            except Exception as e:
                print(f'Error in handling label {label}')
        print(f'handle {dir} success!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', nargs='+', type=str,
                        default=[
                            './traffic_datasets/pcap2csv/iscx-vpn-2016',
                            './traffic_datasets/pcap2csv/iscx-tor-2016',
                            '.traffic_datasets/pcap2csv/ustc-tfc-2016',
                            './traffic_datasets/pcap2csv/ios',
                            './traffic_datasets/pcap2csv/android',
                            './traffic_datasets/pcap2csv/cic',
                        ], help='Data directory')
    parser.add_argument('--save_dir', default='./traffic_datasets/', help='Save directory')

    args = parser.parse_args()

    handle_data(args)


if __name__ == "__main__":
    main()
