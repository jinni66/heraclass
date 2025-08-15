import torch
from tqdm import tqdm
from models.traffic_model import TrafficEncoder
from preprocess.data_loader import traffic_data_loader
from preprocess.traffic_tokenizer import TrafficTokenizer
import json
import os
import argparse

@torch.no_grad()
def extract_traffic_embeddings(config_path: str, data_path: str, save_path: str, device: str = 'cuda:0', traffic_llm_task: str = 'EVD'):

    with open(os.path.join(config_path, 'dataset_config.json')) as f:
        dataset_config = json.load(f)

    with open(os.path.join(config_path, 'model_config.json')) as f:
        model_config = json.load(f)

    timexer_config = model_config['timexer_config']
    traffic_llm_config = model_config['traffic_llm_config']
    decoder_config = model_config['decoder_config']
    traffic_llm_config = model_config['traffic_llm_config']
    traffic_llm_config['task'] = traffic_llm_task

    encoder = TrafficEncoder(
        timexer_config=timexer_config,
        traffic_llm_config=traffic_llm_config,
        device=device
    ).to(device).eval()

    tokenizer = TrafficTokenizer(
        timexer_config=timexer_config,
        traffic_llm_config=traffic_llm_config,
        decoder_config=decoder_config,
        device=device
    )


    dataset_config['common_config']['filter_str'] = 'train'
    dataloader = traffic_data_loader(data_path, **dataset_config)

    all_embeddings = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        labels = batch['labels']

        macro_seq_len = dataset_config['common_config']['macro_seq_len']

        batch_labels = []
        for idx in range(0, len(labels), macro_seq_len):
            macro_label = labels[idx]
            print(f"[Macro {idx // macro_seq_len}] Label: {macro_label}")
            batch_labels.append(macro_label)

        print("=" * 40)


        inputs = tokenizer(
            traffic_llm_data=batch['traffic_llm_data'],
            timexer_data=batch['timexer_data'],
            labels=None
        )
        outputs = encoder(
            timexer_data=inputs['timexer_data'],
            traffic_llm_data=inputs['traffic_llm_data']
        )

        emb = outputs[0].mean(dim=1)
        print(f"[Batch] embedding shape: {emb.shape}")

        all_embeddings.append(emb.cpu())
        all_labels.extend(labels)


    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"[Final] Total embedding shape: {all_embeddings.shape}")
    torch.save({
        "embeddings": all_embeddings,
        "labels": all_labels,
    }, save_path)
    print(f"Saved embeddings to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--traffic_llm_task', type=str, default='EVD')
    args = parser.parse_args()

    extract_traffic_embeddings(
        config_path=args.config_path,
        data_path=args.data_path,
        save_path=args.save_path,
        device=args.device,
        traffic_llm_task=args.traffic_llm_task
    )


if __name__ == '__main__':
    main()