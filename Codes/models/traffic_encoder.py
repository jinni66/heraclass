import copy
import os
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel
import src.TimeXer.models.TimeXer
from preprocess.traffic_tokenizer import TrafficTokenizer
from src.TimeXer.models import TimeXer
from src.TrafficLLM.encoding import load_model as load_trafficllm_peft_model
from utils.common_utils import DictArgs


os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_EVALUATE_OFFLINE'] = "1"


class TrafficEncoder(nn.Module):
    def __init__(self, timexer_config: dict, traffic_llm_config: dict, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.timexer_config = timexer_config
        self.traffic_llm_config = traffic_llm_config

        self.timexer_encoder = self._build_timexer(timexer_config)
        self.traffic_llm_encoder = self._build_traffic_llm(traffic_llm_config)

        self.proj = nn.Sequential(
            nn.Linear(4608, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.LayerNorm(512)
        )

        self.to(self.device)
    def _build_timexer(self, config: dict) -> src.TimeXer.models.TimeXer.Model:
        config = DictArgs(config)
        return TimeXer.Model(config).float().to(self.device)

    def _build_traffic_llm(self, config: dict) -> nn.Module:
        model_config = AutoConfig.from_pretrained(config["model_path"], trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(config["model_path"], config=model_config, trust_remote_code=True)

        task = self.traffic_llm_config.get('task', 'EVD')
        ptuning_path = os.path.join(config["peft_path"], config["peft_set"][task]) if task in config['peft_set'].keys() else None
        peft_model = load_trafficllm_peft_model(model, ptuning_path, self.device)
        return peft_model

    def forward(self, timexer_data, traffic_llm_data):
            _, timexer_enc = self.timexer_encoder(*timexer_data)
            traffic_llm_enc = self.traffic_llm_encoder(**traffic_llm_data, output_hidden_states=True)
            traffic_llm_enc = traffic_llm_enc.hidden_states[-1]

            timexer_enc = timexer_enc.squeeze(1)
            timexer_enc_avg = timexer_enc.sum(dim=-1) / timexer_enc.shape[-1]

            timexer_batch_size = timexer_enc_avg.shape[0]
            timexer_macro_seq_len = self.timexer_config['macro_seq_len']
            timexer_batch_size //= timexer_macro_seq_len
            timexer_enc_avg = timexer_enc_avg.view(timexer_batch_size, timexer_macro_seq_len, -1)

            traffic_llm_enc_avg = (traffic_llm_enc / traffic_llm_enc.shape[0]).sum(dim=0)

            traffic_llm_batch_size = traffic_llm_enc_avg.shape[0]
            traffic_llm_macro_seq_len = self.traffic_llm_config['macro_seq_len']
            traffic_llm_batch_size //= traffic_llm_macro_seq_len
            traffic_llm_enc_avg = traffic_llm_enc_avg.view(traffic_llm_batch_size, traffic_llm_macro_seq_len, -1)

            enc_out = torch.cat((timexer_enc_avg, traffic_llm_enc_avg), -1).to(self.device)
            
            emb = self.proj(enc_out)  

            return (emb,)
