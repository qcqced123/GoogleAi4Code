import gc, random, ast
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from torch.utils.data import Dataset
from torch import Tensor

import configuration
from dataset_class.data_preprocessing import add_special_token, tokenizing


class GoogleAiDataset(Dataset):
    """ Dataset class For Token Classification Pipeline """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.df = df
        self.tokenizer = tokenizing
        self.special_token = add_special_token

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Tensor]:
        self.nb_id = self.df.iloc[item, 0]
        self.cell_id = self.df.iloc[item, 1]
        self.cell_type = self.df.iloc[item, 2]
        self.text = self.df.iloc[item, 3]
        self.ancestor_id = self.df.iloc[item, 5]
        self.pct_rank =




class UPPPMDataset(Dataset):
    """ For Token Classification Task class """
    def __init__(self, cfg, df, is_valid=False):
        super().__init__()
        self.anchor_list = df.anchor.to_numpy()
        self.target_list = df.targets.to_numpy()
        self.context_list = df.context_text.to_numpy()
        self.score_list = df.scores.to_numpy()
        self.id_list = df.ids.to_numpy()
        self.cfg = cfg
        self.is_valid = is_valid

    def tokenizing(self, text: str) -> dict:
        inputs = self.cfg.tokenizer.encode_plus(
            text,
            max_length=self.cfg.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None,
            add_special_tokens=False,
        )
        return inputs

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, idx: int):
        """
        1) make Embedding Shape,
            - Data: [cls]+[anchor]+[sep]+[target]+[tar]+[target]+[tar]...+[tar]+[cpc_text]+[sep]
            - Label: [-1] * self.cfg.max_len, target value의 인덱스 위치에 score_class값 전달
        2) apply data augment
            - shuffle target values
        """
        scores = np.array(ast.literal_eval(self.score_list[idx]))  # len(scores) == target count
        targets = np.array(ast.literal_eval(self.target_list[idx]))

        # Data Augment for train stage: shuffle target value's position index
        if not self.is_valid:
            indices = list(range(len(scores)))
            random.shuffle(indices)
            scores = scores[indices]
            targets = targets[indices]

        text = self.cfg.tokenizer.cls_token + self.anchor_list[idx] + self.cfg.tokenizer.sep_token
        for target in targets:
            text += target + self.cfg.tokenizer.tar_token
        text += self.context_list[idx] + self.cfg.tokenizer.sep_token

        # tokenizing & make label list
        inputs = self.tokenizing(text)
        #target_mask = np.zeros(self.cfg.max_len)
        target_mask = np.zeros(len([token for token in inputs['input_ids'] if token != 0]))
        label = torch.full(
            [len([token for token in inputs['input_ids'] if token != 0])], -1, dtype=torch.float
        )
        # label = torch.full([self.cfg.max_len], -1, dtype=torch.float)
        cnt_tar, cnt_sep, nth_target, prev_i = 0, 0, -1, -1
        for i, input_id in enumerate(inputs['input_ids']):
            if input_id == self.cfg.tokenizer.tar_token_id:
                cnt_tar += 1
                if cnt_tar == len(targets):
                    break
            if input_id == self.cfg.tokenizer.sep_token_id:
                cnt_sep += 1
            if cnt_sep == 1 and input_id not in [self.cfg.tokenizer.pad_token_id, self.cfg.tokenizer.sep_token_id,
                                                 self.cfg.tokenizer.tar_token_id]:
                if (i - prev_i) > 1:
                    nth_target += 1
                label[i] = scores[nth_target]
                target_mask[i] = 1
                prev_i = i

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs, target_mask, label
