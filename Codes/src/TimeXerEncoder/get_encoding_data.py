import argparse
import os
import torch

from src.TimeXer.data_provider.data_factory import data_provider
from src.TimeXer.models import TimeXer


class DictArgs(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def _acquire_device(use_gpu, use_multi_gpu, devices, gpu):
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) if not use_multi_gpu else devices
        device = torch.device('cuda:{}'.format(gpu))
        print('Use GPU: cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device


def _gen_args(is_training=1, root_path='./out/vpn-pcaps-01/', data_path='vpn_ftps_B.csv', seq_len=168, pred_len=24,
              e_layers=3, enc_in=1, patch_len=24, d_model=512, d_ff=512, batch_size=4, target='pkt_size', freq='us'):
    args_dict = {
        # basic config
        'task_name': 'long_term_forecast',
        'is_training': is_training,
        'model_id': 'test',
        'model': 'TimeXer',

        # data loader
        'data': 'data_new',
        'root_path': root_path,
        'data_path': data_path,
        'features': 'S',
        'target': target,
        'freq': freq,
        'timestamp_flag': True,
        'checkpoints': './checkpoints/',

        # forecasting task
        'seq_len': seq_len,
        'label_len': 48,
        'pred_len': pred_len,
        'seasonal_patterns': 'Monthly',
        'inverse': False,

        # inputation task
        'mask_rate': 0.25,

        # anomaly detection task
        'anomaly_ratio': 0.25,

        # model define
        # 'expand': 2,
        # 'd_conv': 4,
        # 'top_k': 5,
        # 'num_kernels': 6,
        'enc_in': enc_in,
        'dec_in': 3,
        'c_out': 1,
        'd_model': d_model,
        'n_heads': 8,
        'e_layers': e_layers,
        'd_layers': 1,
        'd_ff': d_ff,
        'moving_avg': 25,
        'factor': 1,
        'distil': True,
        'dropout': 0.1,
        'embed': 'timeF',
        'activation': 'gelu',
        'output_attention': False,
        'channel_independence': 1,
        'decomp_method': 'moving_avg',
        'use_norm': 1,
        'down_sampling_layers': 0,
        'down_sampling_window': 1,
        'down_sampling_method': None,
        'seg_len': 48,

        # optimization
        'num_workers': 10,
        'itr': 1,
        'train_epochs': 2,
        'batch_size': batch_size,
        'patience': 3,
        'learning_rate': 0.0001,
        'des': 'Timexer-MS',
        'loss': 'MSE',
        'lradj': 'type1',
        'use_amp': False,

        # GPU
        'use_gpu': True,
        'gpu': 0,
        'use_multi_gpu': False,
        'devices': '0,1,2,3',

        # de-stationary projector params
        'p_hidden_dims': [128, 128],
        'p_hidden_layers': 2,

        # metrics (dtw)
        'use_dtw': False,

        # Augmentation
        'augmentation_ratio': 0,
        'seed': 2,
        'jitter': False,
        'scaling': False,
        'permutation': False,
        'randompermutation': False,
        'magwarp': False,
        'timewarp': False,
        'windowslice': False,
        'windowwarp': False,
        'rotation': False,
        'spawner': False,
        'dtwwarp': False,
        'shapedtwwarp': False,
        'wdba': False,
        'discdtw': False,
        'discsdtw': False,
        'extra_tag': '',

        # TimeXer
        'patch_len': patch_len,
        'need_encoding_data': True

    }
    args = DictArgs(args_dict)
    return args


def get_encoding_data(is_training=1, root_path='./out/vpn-pcaps-01/', data_path='vpn_ftps_B.csv', seq_len=168,
                      pred_len=24, e_layers=3, enc_in=1, patch_len=24, d_model=512, d_ff=512, batch_size=4,
                      target='pkt_size', freq='us'):
    args = _gen_args(is_training, root_path, data_path, seq_len, pred_len, e_layers, enc_in, patch_len, d_model, d_ff,
                     batch_size, target, freq)
    # device = _acquire_device(args.use_gpu, args.use_multi_gpu, args.devices, args.gpu)
    device = 'cpu'
    args.used_col = ['date', 'pkt_size']
    data_set, data_loader = data_provider(args, 'test')
    print('dataset size: {}'.format(len(data_set)))
    enc_out_list = []

    model = TimeXer.Model(args).float().to(device)
    with torch.no_grad():
        model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            _, enc_out = model(batch_x, batch_x_mark, None, None)
            print(enc_out)
            print(enc_out.shape)
            enc_out_list.append(enc_out)
    return enc_out_list


if __name__ == '__main__':
    get_encoding_data()
