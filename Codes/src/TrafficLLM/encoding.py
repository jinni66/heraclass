from transformers import AutoTokenizer, AutoModel, AutoConfig
import fire
import torch
import json
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(model, ptuning_path, device='cpu'):
    if ptuning_path is not None:
        prefix_state_dict = torch.load(
            os.path.join(ptuning_path, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    if device is not None:
        model = model.half().to(device)
    else:
        model = model.half().cuda()
    model.transformer.prefix_encoder.float()

    return model


def main(config, traffic_data: str = None, **kwargs):
    with open(config, "r", encoding="utf-8") as fin:
        config = json.load(fin)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(config["model_path"], trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(config["model_path"], config=model_config, trust_remote_code=True)

    task = "EVD"
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"][task])
    model_downstream = load_model(model, ptuning_path)

    model_downstream = model_downstream.eval()

    #inputs_traffic = tokenizer(traffic_data, return_tensors="pt").to(model_downstream.device)
    inputs_traffic = tokenizer(
        traffic_data,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt"
    ).to(model_downstream.device)
    with torch.no_grad():
        encoder_outputs_traffic = model_downstream(**inputs_traffic, output_hidden_states=True)

    embeddings_traffic = encoder_outputs_traffic.hidden_states[-1]
    print(embeddings_traffic)


if __name__ == "__main__":
    fire.Fire(main)

