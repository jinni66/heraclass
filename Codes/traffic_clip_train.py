import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import clip
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class TrafficDataset(Dataset):
    def __init__(self, json_path, max_length=77):
        self.data = []
        self.labels = []
        self.max_length = max_length
        self.label_to_idx = {}
        self.idx_to_label = {}

        with open(json_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line.strip())
                input_text = entry["input"][:max_length]
                self.data.append(input_text)
                label = entry["output"]
                if label not in self.label_to_idx:
                    idx = len(self.label_to_idx)
                    self.label_to_idx[label] = idx
                    self.idx_to_label[idx] = label
                self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)
    tokenized_inputs = clip.tokenize(list(inputs), truncate=True)
    return tokenized_inputs, torch.tensor(labels)

def build_label_prompts(idx_to_label, templates=None):
    if templates is None:
        templates = [
            "This is a traffic about {}.",
            "It is likely a {} network flow.",
            "Traffic related to {} application.",
            "The flow behavior corresponds to {}.",
            "An example of {} internet traffic."
        ]
    prompts = []
    for idx in sorted(idx_to_label.keys()):
        label = idx_to_label[idx].replace("_", " ")
        label_prompts = [template.format(label) for template in templates]
        prompts.append(label_prompts)
    return prompts

def encode_label_features(prompts, model):
    all_features = []
    for prompt_set in prompts:
        tokenized = clip.tokenize(prompt_set).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(tokenized)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            avg_feature = embeddings.mean(dim=0, keepdim=True)
            all_features.append(avg_feature)
    return torch.cat(all_features, dim=0)

def evaluate(model, dataloader, label_prompts):
    model.eval()
    correct = 0
    total = 0
    label_features = encode_label_features(label_prompts, model)

    with torch.no_grad():
        for tokenized_inputs, text_labels in dataloader:
            tokenized_inputs = tokenized_inputs.to(device)
            text_labels = text_labels.to(device)

            input_features = model.encode_text(tokenized_inputs)
            input_features = input_features / input_features.norm(dim=-1, keepdim=True)
            logits = input_features @ label_features.T

            preds = logits.argmax(dim=-1)
            correct += (preds == text_labels).sum().item()
            total += text_labels.size(0)

    acc = correct / total
    return acc

model, preprocess = clip.load("ViT-B/32", device)
model = model.float()

for name, param in model.named_parameters():
    param.requires_grad = "transformer" in name

dataset = TrafficDataset("ios_train.json")

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=30)

prompt_templates = [
    "This is a traffic about {}.",
    "It is likely a {} network flow.",
    "Traffic related to {} application.",
    "The flow behavior corresponds to {}.",
    "An example of {} internet traffic."
]
label_prompts = build_label_prompts(dataset.idx_to_label, prompt_templates)

best_val_acc = 0.0
epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    label_features = encode_label_features(label_prompts, model)

    for tokenized_inputs, text_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        tokenized_inputs = tokenized_inputs.to(device)
        text_labels = text_labels.to(device)

        optimizer.zero_grad()

        input_features = model.encode_text(tokenized_inputs)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        logits = input_features @ label_features.T

        loss = F.cross_entropy(logits, text_labels, label_smoothing=0.1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    val_acc = evaluate(model, val_loader, label_prompts)

    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print(f"✅ Best model saved at Epoch {epoch+1} with Val Acc: {val_acc:.4f}")
torch.save(model.state_dict(), "ios_label_model.pt")
