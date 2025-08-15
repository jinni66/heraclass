# TODO: do tokenization
import torch
from transformers import AutoTokenizer


class TrafficTokenizer:
    def __init__(self,
                 timexer_config: dict,
                 traffic_llm_config: dict,
                 decoder_config: dict,
                 device='cpu',
                 **kwargs):
        self.timexer_config = timexer_config
        self.traffic_llm_config = traffic_llm_config
        self.decoder_config = decoder_config
        self.device = device
        self.traffic_llm_tokenizer = AutoTokenizer.from_pretrained(
            self.traffic_llm_config["model_path"],
            trust_remote_code=True
        )
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(
            self.decoder_config['model_name_or_path'],
            trust_remote_code=True
        )
        self.decoder_start_token_id = 0  # refer to t5 config

    def __call__(self, traffic_llm_data, timexer_data, labels=None):
        # traffic llm
        inputs_traffic = self.traffic_llm_tokenizer(
            traffic_llm_data,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        ).to(self.device)

        # timexer
        batch_x, batch_y, batch_x_mark, batch_y_mark = timexer_data
        batch_x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_x.float().to(self.device)
        batch_y_mark = batch_x_mark.float().to(self.device)

        macro_seq_len = self.traffic_llm_config['macro_seq_len']
        batch_size = len(traffic_llm_data) // macro_seq_len

        # decoder
        if labels is not None:
            batch_labels = []
            for idx in range(0, len(labels), macro_seq_len):
                _label = labels[idx]
                _labels = labels[idx+1 : idx + macro_seq_len]
                if any(l != _label for l in _labels):
                    raise ValueError("labels in a macro sequence are not the same")
                batch_labels.append(_label)
            decoder_label_ids = self.decoder_tokenizer(
                batch_labels,
                return_tensors='pt'
            ).input_ids.to(self.device)
            decoder_input_ids = self._shift_right(decoder_label_ids)
        else:
            decoder_input_ids = [[self.decoder_start_token_id]] * batch_size
            decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long).to(self.device)
            decoder_label_ids = None

        return {
            'traffic_llm_data': inputs_traffic,
            'timexer_data': (batch_x, batch_y, batch_x_mark, batch_y_mark),
            'decoder_input_ids': decoder_input_ids,
            'labels': decoder_label_ids
        }

    def _shift_right(self, input_ids):
        pad_token_id = self.decoder_tokenizer.pad_token_id

        if self.decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
                "See T5 docs for more information."
            )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
