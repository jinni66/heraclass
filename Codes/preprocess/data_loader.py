from torch.utils.data import DataLoader

from preprocess.dataset import TrafficDataset


def traffic_data_loader(data_path, common_config, timexer_config, traffic_llm_config):
    batch_size = common_config['batch_size']
    macro_seq_len = common_config['macro_seq_len']
    dataset = TrafficDataset(data_path, common_config, timexer_config, traffic_llm_config)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size * macro_seq_len,
        drop_last=True
    )
    return data_loader

