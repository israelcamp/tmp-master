import torch
from transformers import DebertaV2ForTokenClassification, DebertaV2Config


class AbstractTransformersEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int = 100, config_dict: dict = None):
        super().__init__()
        self.vocab_size = vocab_size
        config_dict = self._get_config_dict(config_dict)
        config = DebertaV2Config(**config_dict)
        self.encoder = DebertaV2ForTokenClassification(config)

    def _get_config_dict(self, config_dict):
        base_config_dict = {
            "model_type": "deberta-v2",
            "architectures": ["DebertaV2ForTokenClassification"],
            "num_labels": self.vocab_size,
            "model_type": "deberta-v2",
            "attention_probs_dropout_prob": 0.25,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.25,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 768,  # 3072,
            "max_position_embeddings": 512,
            "relative_attention": True,
            "position_buckets": 256,  # TODO: Maybe less?
            "norm_rel_ebd": "layer_norm",
            "share_att_key": True,
            "pos_att_type": "p2c|c2p",
            "layer_norm_eps": 1e-7,
            "max_relative_positions": -1,
            "position_biased_input": True,
            "num_attention_heads": 8,
            "num_hidden_layers": 3,
            "type_vocab_size": 0,
            "pad_token_id": 1,
            "vocab_size": self.vocab_size,
        }
        base_config_dict.update(config_dict)
        return base_config_dict

    def forward(self, image_embeddings, attention_mask=None):
        outputs = self.encoder(
            inputs_embeds=image_embeddings, attention_mask=attention_mask
        )
        return outputs.logits


class TransformersEncoderBase(AbstractTransformersEncoder):
    def __init__(self, vocab_size: int = 100):
        config_dict = {
            "hidden_size": 768,
            "intermediate_size": 768,
            "num_attention_heads": 6,
            "num_hidden_layers": 6,
        }
        super().__init__(vocab_size=vocab_size, config_dict=config_dict)


class TransformersEncoderSmall(AbstractTransformersEncoder):
    def __init__(self, vocab_size: int = 100):
        config_dict = {
            "hidden_size": 512,
            "intermediate_size": 768,
            "num_attention_heads": 8,
            "num_hidden_layers": 3,
        }
        super().__init__(vocab_size=vocab_size, config_dict=config_dict)
