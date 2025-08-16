import os

import pandas as pd
from torch.utils.data import Dataset
from src.TimeXer.utils.timefeatures import time_features
from utils.common_utils import get_files


class TrafficDataset(Dataset):
    def __init__(self, data_path: str,
                 common_config: dict,
                 timexer_config: dict,
                 traffic_llm_config: dict):
        self.data_path = data_path
        self.common_config = common_config
        self.traffic_llm_config = traffic_llm_config
        self.timexer_config = timexer_config

        self.seq_len = self.common_config['seq_len']
        self.target = self.common_config['target']
        self.filter_str = self.common_config['filter_str'] if 'filter_str' in self.common_config else None

        self.timeenc = self.timexer_config['timeenc']
        self.timestamp_flag = self.timexer_config.get('timestamp_flag', False)
        self.freq = self.timexer_config['freq']
        self.used_col = self.timexer_config.get('used_col')
        self.features = self.timexer_config['features']
        self.timexer_target = self.timexer_config['target']

        self.traffic_data_col_mapping = self.traffic_llm_config['traffic_data_col_mapping']

        self.__read_data__()

    def __read_data__(self):
        if os.path.isdir(self.data_path):
            file_list = get_files(self.data_path, self.filter_str)
        else:
            file_list = [self.data_path]

        df_list = []
        for file_name in file_list:
            df_list.append(pd.read_csv(file_name))

        df_raw = pd.concat(df_list)
        self._handle_timexer_data(df_raw)
        self._handle_traffic_llm_data(df_raw)
        self._handle_traffic_target(df_raw)

    def _handle_timexer_data(self, df_raw):
        if self.used_col is not None:
            df_raw = df_raw[self.used_col]

        cols = list(df_raw.columns)
        cols.remove(self.timexer_target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.timexer_target]]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.timexer_target]]
        else:
            raise Exception('Unknown features')

        data = df_data.values

        df_stamp = df_raw[['date']]
        if self.timestamp_flag:
            df_stamp['date'] = pd.to_datetime(df_stamp.date, unit='s')
        else:
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise Exception(f'TimeEnc {self.timeenc} error!')

        self.timexer_data_x = data
        self.timexer_data_y = data

        self.data_stamp = data_stamp

    def _handle_traffic_llm_data(self, df_raw):
        cols = self.traffic_data_col_mapping.values()
        df_data = df_raw[cols]

        self.traffic_data = df_data

    def _handle_traffic_target(self, df_raw):
        df_data = df_raw[self.target].values

        self.target_data = df_data

    # def _get_actual_seq_len(self, index):
    #     target = self.target_data[index]
    #     for i in range(index, index+self.seq_len):
    #         if target != self.target_data[i]:
    #             return i - index
    #     return self.seq_len

    def __getitem__(self, index):
        # seq_len = self._get_actual_seq_len(index)

        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.timexer_data_x[s_begin: s_end]
        seq_y = seq_x
        seq_x_mark = self.data_stamp[s_begin: s_end]
        seq_y_mark = seq_x_mark

        traffic_data_raw = self.traffic_data[s_begin: s_end]
        traffic_data = ''
        for _, row in traffic_data_raw.iterrows():
            for key, value in self.traffic_data_col_mapping.items():
                traffic_data += f'{key}: {row[value]} '

        target = self.target_data[s_begin: s_end]
        target = list(set(list(target)))
        target = ','.join(target)

        return {
            'timexer_data': (seq_x, seq_y, seq_x_mark, seq_y_mark),
            'traffic_llm_data': traffic_data,
            'labels': target
        }

    def __len__(self):
        return len(self.timexer_data_x) - self.seq_len + 1
