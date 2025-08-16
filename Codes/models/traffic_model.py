import copy
import os
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, \
    InfNanRemoveLogitsProcessor, LogitsProcessorList
from transformers.generation import SampleEncoderDecoderOutput
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, CausalLMOutputWithPast

import src.TimeXer.models.TimeXer
from preprocess.traffic_tokenizer import TrafficTokenizer
from src.TimeXer.models import TimeXer
from src.TrafficLLM.encoding import load_model as load_trafficllm_peft_model
from transformers.models.t5.modeling_t5 import T5Stack

from utils.common_utils import DictArgs


os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_EVALUATE_OFFLINE'] = "1"


class TrafficEncoder(nn.Module):
    def __init__(self,
                 timexer_config: dict,
                 traffic_llm_config: dict,
                 device='cuda:0'):
        super().__init__()

        self.device = device

        self.timexer_config = timexer_config
        self.timexer_encoder = self._build_timexer(timexer_config)

        self.traffic_llm_config = traffic_llm_config
        self.traffic_llm_encoder = self._build_traffic_llm(traffic_llm_config)

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
        return (enc_out,)


class TrafficDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        model_name_or_path = config["model_name_or_path"]
        encoder_output_dim = config["encoder_output_dim"]

        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.t5_decoder = t5_model.decoder
        self.lm_head = t5_model.lm_head
        self.t5_dim = t5_model.config.d_model

        self.input_proj = nn.Linear(encoder_output_dim, self.t5_dim, bias=False)

    def forward(self,
                input_ids,
                attention_mask,
                encoder_hidden_states,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):
        projected_encoder_hidden_states = self.input_proj(encoder_hidden_states)
        decoder_outputs = self.t5_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=projected_encoder_hidden_states,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        if self.t5_decoder.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + (encoder_hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )


class TrafficModel(nn.Module):
    def __init__(self,
                 timexer_config: dict,
                 traffic_llm_config: dict,
                 decoder_config: dict,
                 device='cpu',
                 ):
        super().__init__()
        self.device = device
        self.encoder = TrafficEncoder(
            timexer_config=timexer_config,
            traffic_llm_config=traffic_llm_config,
            device=self.device
        )
        self.decoder = TrafficDecoder(decoder_config).to(self.device)

    def forward(
            self,
            timexer_data,
            traffic_llm_data,
            decoder_input_ids,
            encoder_outputs = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                timexer_data = timexer_data,
                traffic_llm_data = traffic_llm_data,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=None,
                attentions=None,
            )

        encoder_hidden_states = encoder_outputs[0]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return decoder_outputs

    def generate(self,
                 timexer_data,
                 traffic_llm_data,
                 decoder_input_ids,
                 max_length: int = 5,
                 return_dict: Optional[bool] = None,
                 labels = None,
                 greedy=False):

        encoder_outputs = self.encoder(
            timexer_data=timexer_data,
            traffic_llm_data=traffic_llm_data
        )

        logits_processor = LogitsProcessorList()
        logits_processor.append(InfNanRemoveLogitsProcessor())


        past_key_values = None
        for step in range(max_length):
            decoder_outputs = self(
                timexer_data=timexer_data,
                traffic_llm_data=traffic_llm_data,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                inputs_embeds=None,
                past_key_values=past_key_values,
                labels=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            next_token_logits = decoder_outputs.logits[:, -1, :]
            next_token_scores = logits_processor(decoder_input_ids, next_token_logits)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            if greedy:
                next_tokens = probs.argmax(dim=-1).unsqueeze(1)
            else:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            decoder_input_ids = torch.cat((decoder_input_ids, next_tokens[:, None]), dim=-1)
            past_key_values = decoder_outputs.past_key_values

        if return_dict:
            return SampleEncoderDecoderOutput(
                sequences=decoder_input_ids,
                scores=None,
                encoder_attentions=None,
                encoder_hidden_states=None,
                decoder_attentions=None,
                cross_attentions=None,
                decoder_hidden_states=None,
            )
        else:
            return decoder_input_ids


