import os
import json
import torch
import clip
from tqdm import tqdm

from models.traffic_encoder import TrafficEncoder
from preprocess.traffic_tokenizer import TrafficTokenizer
from preprocess.data_loader import traffic_data_loader


class CombinedModel(torch.nn.Module):
    def __init__(self, timexer_config, traffic_llm_config, device):
        super().__init__()
        self.device = device
        self.traffic_encoder = TrafficEncoder(timexer_config, traffic_llm_config, device)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()


@torch.no_grad()
def inference(model_path, config_path, data_path, output_path, traffic_llm_task="EVD", device="cuda"):
    with open(os.path.join(config_path, "model_config.json")) as f:
        model_config = json.load(f)
    with open(os.path.join(config_path, "dataset_config.json")) as f:
        dataset_config = json.load(f)

    timexer_config = model_config["timexer_config"]
    traffic_llm_config = model_config["traffic_llm_config"]
    decoder_config = model_config["decoder_config"]
    traffic_llm_config["task"] = traffic_llm_task

    macro_seq_len = dataset_config["common_config"]["macro_seq_len"]

    model = CombinedModel(timexer_config, traffic_llm_config, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset_config['common_config']['filter_str'] = "test"
    test_loader = traffic_data_loader(data_path, **dataset_config)
    tokenizer = TrafficTokenizer(timexer_config, traffic_llm_config, decoder_config, device)

    all_labels = set()
    filtered_batches = []
    for batch in test_loader:
        labels = batch['labels']
        macro_labels = [labels[i] for i in range(0, len(labels), macro_seq_len)]
        if all(',' not in lbl for lbl in macro_labels):
            all_labels.update(macro_labels)
            filtered_batches.append(batch)

    label_list = sorted(list(all_labels))
    label2idx = {label: idx for idx, label in enumerate(label_list)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    label_texts = [f"This is a traffic about {label}" for label in label_list]
    tokenized = clip.tokenize(label_texts).to(device)
    label_features = model.clip_model.encode_text(tokenized)
    label_features = label_features / (label_features.norm(dim=-1, keepdim=True) + 1e-8)
    label_features = label_features.float()

    predictions = []

    for batch in tqdm(filtered_batches, desc="🔍 "):
        labels = batch['labels']
        macro_labels = [labels[j] for j in range(0, len(labels), macro_seq_len)]

        inputs = tokenizer(
            traffic_llm_data=batch['traffic_llm_data'],
            timexer_data=batch['timexer_data'],
            labels=None
        )

        traffic_features = model.traffic_encoder(
            timexer_data=inputs['timexer_data'],
            traffic_llm_data=inputs['traffic_llm_data']
        )[0].mean(dim=1)
        traffic_features = traffic_features / (traffic_features.norm(dim=-1, keepdim=True) + 1e-8)

        sim_logits = traffic_features @ label_features.T
        pred_indices = sim_logits.argmax(dim=1).tolist()
        pred_labels = [idx2label[idx] for idx in pred_indices]

        for true_label, pred_label in zip(macro_labels, pred_labels):
            predictions.append({
                "true_label": true_label,
                "pred_label": pred_label
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--traffic_llm_task', type=str, default="EVD")
    parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()

    inference(
        model_path=args.model_path,
        config_path=args.config_path,
        data_path=args.data_path,
        output_path=args.output_path,
        traffic_llm_task=args.traffic_llm_task,
        device=args.device
    )
