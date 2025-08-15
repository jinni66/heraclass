import json
import torch
import clip
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _ = clip.load("ViT-B/32", device)
model = model.float()
model.eval()

model.load_state_dict(torch.load("best_model.pt", map_location=device))

test_json_path = "ios_test.json"
test_data = []
true_labels = []
with open(test_json_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line.strip())
        test_data.append(entry["input"])
        true_labels.append(entry["output"])

all_labels = sorted(list(set(true_labels)))

def build_label_prompts(labels, templates=None):
    if templates is None:
        templates = [
            "This is a traffic about {}.",
            "It is likely a {} network flow.",
            "Traffic related to {} application.",
            "The flow behavior corresponds to {}.",
            "An example of {} internet traffic."
        ]
    prompts = []
    for label in labels:
        label_clean = label.replace("_", " ")
        label_prompts = [template.format(label_clean) for template in templates]
        prompts.append(label_prompts)
    return prompts

def encode_label_features(prompts):
    features = []
    for prompt_set in prompts:
        tokens = clip.tokenize(prompt_set).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            mean_embedding = embeddings.mean(dim=0, keepdim=True)
            features.append(mean_embedding)
    return torch.cat(features, dim=0)

label_prompts = build_label_prompts(all_labels)
label_features = encode_label_features(label_prompts)

pred_labels = []
batch_size = 64
MAX_CHAR_LEN = 300

with torch.no_grad():
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch_texts = test_data[i:i + batch_size]
        tokenized = clip.tokenize(batch_texts, truncate=True).to(device)
        text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = text_features @ label_features.T
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        pred_labels.extend([all_labels[p] for p in preds])

def load_specified_categories(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        return {cat.strip() for cat in line.split(",") if cat.strip()}

specified_categories_file = "training/ios_label.txt"
specified_categories = load_specified_categories(specified_categories_file)

output_lines = []

top1_acc = accuracy_score(true_labels, pred_labels)
output_lines.append(f"Top-1 Accuracy: {top1_acc:.4f}\n")

class_report = classification_report(true_labels, pred_labels, digits=4)
output_lines.append("Detailed classification report:\n")
output_lines.append(class_report)

precision_all, recall_all, f1_all, support_all = precision_recall_fscore_support(
    true_labels, pred_labels, labels=all_labels, zero_division=0
)
metrics_per_class = {
    label: {"precision": p, "recall": r, "f1": f1}
    for label, p, r, f1 in zip(all_labels, precision_all, recall_all, f1_all)
}

specified_metrics = [metrics_per_class[label] for label in all_labels if label in specified_categories]

output_lines.append("\n:")
output_lines.append(", ".join(specified_categories) + "\n")

if specified_metrics:
    macro_precision = sum(m["precision"] for m in specified_metrics) / len(specified_metrics)
    macro_recall = sum(m["recall"] for m in specified_metrics) / len(specified_metrics)
    macro_f1 = sum(m["f1"] for m in specified_metrics) / len(specified_metrics)

    output_lines.append("[Macro Average]")
    output_lines.append(f"Precision: {macro_precision:.4f}")
    output_lines.append(f"Recall:    {macro_recall:.4f}")
    output_lines.append(f"F1-score:  {macro_f1:.4f}\n")

    output_lines.append("[labels]")
    for label in all_labels:
        if label in specified_categories:
            m = metrics_per_class[label]
            output_lines.append(f"{label:25s} | Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}")
else:
    output_lines.append("NAN")

output_file = "eval_result.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")
