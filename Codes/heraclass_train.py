import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip
import argparse
import time

from models.traffic_encoder import TrafficEncoder
from preprocess.traffic_tokenizer import TrafficTokenizer
from preprocess.data_loader import traffic_data_loader

class CombinedModel(torch.nn.Module):
    def __init__(self, timexer_config, traffic_llm_config, device):
        super().__init__()
        self.device = device
        self.traffic_encoder = TrafficEncoder(timexer_config, traffic_llm_config, device)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.train()

def train_clip_softmax(config_path, data_path, save_path, log_path, traffic_llm_task="EVD", device="cuda", num_epochs=10):
    with open(os.path.join(config_path, "model_config.json")) as f:
        model_config = json.load(f)
    with open(os.path.join(config_path, "dataset_config.json")) as f:
        dataset_config = json.load(f)

    timexer_config = model_config["timexer_config"]
    traffic_llm_config = model_config["traffic_llm_config"]
    decoder_config = model_config["decoder_config"]
    traffic_llm_config["task"] = traffic_llm_task

    macro_seq_len = dataset_config['common_config']['macro_seq_len']
    model = CombinedModel(timexer_config, traffic_llm_config, device).to(device)

    dataset_config['common_config']['filter_str'] = "train"
    train_loader = traffic_data_loader(data_path, **dataset_config)
    tokenizer = TrafficTokenizer(timexer_config, traffic_llm_config, decoder_config, device)

    all_labels = set()
    filtered_batches = []
    for batch in train_loader:
        labels = batch['labels']
        macro_labels = [labels[i] for i in range(0, len(labels), macro_seq_len)]
        if all(',' not in lbl for lbl in macro_labels):
            all_labels.update(macro_labels)
            filtered_batches.append(batch)


    label_list = sorted(list(all_labels))
    label2idx = {label: idx for idx, label in enumerate(label_list)}

    label_texts = [f"This is a traffic about {label}" for label in label_list]
    tokenized = clip.tokenize(label_texts).to(device)
    with torch.no_grad():
        label_features = model.clip_model.encode_text(tokenized)
    eps = 1e-8
    label_features = label_features / (label_features.norm(dim=-1, keepdim=True) + eps)
    label_features = label_features.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    total_steps = num_epochs * len(filtered_batches)
    warmup_steps = int(0.2 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / (warmup_steps + 1), 1.0))

    resume_path = os.path.join(os.path.dirname(save_path), "checkpoint_latest.pth")
    best_model_path = os.path.join(os.path.dirname(save_path), "checkpoint_best.pth")
    start_epoch = 0
    best_acc = 0.0
    patience = 3
    no_improve_epochs = 0

    if os.path.exists(resume_path):
        print(f"checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1

    temp = 0.07

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    avg_loss = None
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for i, batch in enumerate(tqdm(filtered_batches, desc=f"[Epoch {epoch + 1}]")):
            labels = batch['labels']
            macro_labels = [labels[j] for j in range(0, len(labels), macro_seq_len)]
            label_indices = torch.tensor([label2idx[lbl] for lbl in macro_labels]).to(device)

            inputs = tokenizer(
                traffic_llm_data=batch['traffic_llm_data'],
                timexer_data=batch['timexer_data'],
                labels=None
            )

            traffic_features = model.traffic_encoder(
                timexer_data=inputs['timexer_data'],
                traffic_llm_data=inputs['traffic_llm_data']
            )[0].mean(dim=1)
            traffic_features = traffic_features / (traffic_features.norm(dim=-1, keepdim=True) + eps)
            traffic_features = traffic_features.float()

            logits = traffic_features @ label_features.T  # [B, C]
            loss = F.cross_entropy(logits / temp, label_indices)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️NaN/Inf batch {i}")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in filtered_batches:
                labels = batch['labels']
                macro_labels = [labels[j] for j in range(0, len(labels), macro_seq_len)]
                label_indices = torch.tensor([label2idx[lbl] for lbl in macro_labels]).to(device)

                inputs = tokenizer(
                    traffic_llm_data=batch['traffic_llm_data'],
                    timexer_data=batch['timexer_data'],
                    labels=None
                )

                traffic_features = model.traffic_encoder(
                    timexer_data=inputs['timexer_data'],
                    traffic_llm_data=inputs['traffic_llm_data']
                )[0].mean(dim=1)
                traffic_features = traffic_features / (traffic_features.norm(dim=-1, keepdim=True) + eps)

                sim_logits = traffic_features @ label_features.T
                preds = sim_logits.argmax(dim=1)
                correct += (preds == label_indices).sum().item()
                total += preds.size(0)

        top1_acc = 100.0 * correct / max(total, 1)
        print(f"[Epoch {epoch + 1}] Top-1 Accuracy: {top1_acc:.2f}%")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, resume_path)

        if top1_acc > best_acc:
            best_acc = top1_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            break

    total_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    print(f"🕒: {total_time:.2f}s")
    print(f"💾: {peak_memory:.2f} MB")
    print(f"✅: {save_path}")

    with open(log_path, "a", encoding="utf-8") as f_log:
        f_log.write(f"{total_time:.2f}s\n")
        f_log.write(f"{peak_memory:.2f}MB\n")
        if avg_loss is not None:
            f_log.write(f"{avg_loss:.4f}\n")
        else:
            f_log.write("N/A\n")
        f_log.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
        f_log.write(f"Best Top-1 Accuracy: {best_acc:.2f}%\n")
        f_log.write(f"{save_path}\n")
        f_log.write(f"{best_model_path}\n")
        f_log.write(f"{label2idx}\n")
        f_log.write("-" * 40 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--traffic_llm_task', type=str, default='EVD')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()

    train_clip_softmax(
        config_path=args.config_path,
        data_path=args.data_path,
        save_path=args.save_path,
        log_path=args.log_path,
        traffic_llm_task=args.traffic_llm_task,
        device=args.device,
        num_epochs=args.num_epochs
    )
