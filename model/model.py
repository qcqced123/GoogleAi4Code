import torch
import torch.nn as nn
import model.pooling as pooling
from torch import Tensor
from transformers import AutoConfig, AutoModel

import configuration
from model.model_utils import freeze, reinit_topk


class GoogleAi4CodeModel(nn.Module):
    """
    Model class for pair-wise(Margin Ranking), dict-wise(Multiple Negative Ranking) pipeline with DeBERTa-V3-Large
    Apply Pooling & Fully Connected Layer for each unique cell in one notebook_id
    Args:
        cfg: configuration.CFG
    Reference:
        https://www.kaggle.com/competitions/AI4Code/discussion/368997
    """
    def __init__(self, cfg: configuration.CFG):
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 1)
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=False
            )  # load student model's weight: it already has fc layer, so need to init fc layer later

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict, position_list: list[list[list[int, int]]]) -> list[list[Tensor]]:
        outputs = self.feature(inputs)
        feature = outputs.last_hidden_state
        pred_rank = []
        for i in range(self.cfg.batch_size):
            """ Apply Pooling & Fully Connected Layer for each unique cell in batch (one notebook_id) """
            instance_rank = []
            for idx in range(len(position_list[i])):
                src, end = position_list[i][idx]
                embedding = self.pooling(feature[i, src:end+1, :])  # maybe don't need mask
                logit = self.fc(embedding)
                instance_rank.append(logit)
            pred_rank.append(instance_rank)
        return pred_rank
