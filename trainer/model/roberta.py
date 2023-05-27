import torch
from transformers import RobertaForTokenClassification, RobertaConfig


class RobertaEncoderBase(torch.nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()

        config_dict = {
            "architectures": ["RobertaForTokenClassification"],
            "num_labels": vocab_size,
            "attention_probs_dropout_prob": 0.25,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.25,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 768,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 514,
            "model_type": "roberta",
            "num_attention_heads": 6,
            "num_hidden_layers": 6,
            "pad_token_id": 1,
            "type_vocab_size": 1,
            "vocab_size": vocab_size,
        }
        config = RobertaConfig(**config_dict)
        self.encoder = RobertaForTokenClassification(config)

    def forward(self, image_embeddings, attention_mask=None):
        outputs = self.encoder(
            inputs_embeds=image_embeddings, attention_mask=attention_mask
        )
        return outputs.logits


class RobertaEncoderSmall(torch.nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()

        config_dict = {
            "architectures": ["RobertaForTokenClassification"],
            "num_labels": vocab_size,
            "attention_probs_dropout_prob": 0.25,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.25,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 768,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 514,
            "model_type": "roberta",
            "num_attention_heads": 8,
            "num_hidden_layers": 3,
            "pad_token_id": 1,
            "type_vocab_size": 1,
            "vocab_size": vocab_size,
        }
        config = RobertaConfig(**config_dict)
        self.encoder = RobertaForTokenClassification(config)

    def forward(self, image_embeddings, attention_mask=None):
        outputs = self.encoder(
            inputs_embeds=image_embeddings, attention_mask=attention_mask
        )
        return outputs.logits
