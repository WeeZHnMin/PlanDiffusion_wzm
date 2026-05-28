"""
图结构生成模型，支持两种模式：
  pretrain : 纯 GPT-2 自回归（无文本条件）
  sft      : BERT 编码器 + GPT-2 解码器（cross-attention）
"""

import torch
import torch.nn as nn
from transformers import BertModel, GPT2Config, GPT2LMHeadModel

BERT_HIDDEN = 768


class GraphModel(nn.Module):
    """预训练阶段：纯 GPT-2 自回归，无文本条件。"""

    def __init__(self, gpt2_cfg: dict):
        super().__init__()
        config = GPT2Config(**gpt2_cfg)
        self.gpt2 = GPT2LMHeadModel(config)

    def forward(self, token_ids, token_mask, labels=None):
        return self.gpt2(
            input_ids=token_ids,
            attention_mask=token_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, bos_id, eos_id, pad_id, batch_size=1,
                 max_length=200, do_sample=True, top_k=50, device='cpu'):
        init = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        return self.gpt2.generate(
            input_ids=init,
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )


class TextConditionedGraphModel(nn.Module):
    """SFT 阶段：BERT 编码 prompt，GPT-2 通过 cross-attention 条件生成。"""

    def __init__(self, bert_path: str, gpt2_cfg: dict,
                 pretrained_gpt2_path: str = None):
        super().__init__()

        # 文本编码器
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_proj = nn.Linear(BERT_HIDDEN, gpt2_cfg['n_embd'])

        # 图结构解码器（开启 cross-attention）
        config = GPT2Config(
            **gpt2_cfg,
            add_cross_attention=True,
            is_decoder=True,
        )
        self.gpt2 = GPT2LMHeadModel(config)

        # 加载预训练 GPT-2 权重（只加载 gpt2 子模块部分）
        if pretrained_gpt2_path is not None:
            state = torch.load(pretrained_gpt2_path, map_location='cpu')
            # 预训练保存的是 GraphModel 的 state_dict，key 前缀是 "gpt2."
            gpt2_state = {k[len('gpt2.'):]: v for k, v in state.items()
                          if k.startswith('gpt2.')}
            missing, unexpected = self.gpt2.load_state_dict(gpt2_state, strict=False)
            print(f'GPT-2 预训练权重加载完成  missing={len(missing)}  unexpected={len(unexpected)}')

    def encode_text(self, text_ids, text_mask):
        bert_out = self.bert(input_ids=text_ids, attention_mask=text_mask)
        return self.bert_proj(bert_out.last_hidden_state)

    def forward(self, token_ids, token_mask, text_ids, text_mask, labels=None):
        encoder_hidden = self.encode_text(text_ids, text_mask)
        return self.gpt2(
            input_ids=token_ids,
            attention_mask=token_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=text_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, text_ids, text_mask, bos_id, eos_id, pad_id,
                 max_length=200, do_sample=True, top_k=50):
        encoder_hidden = self.encode_text(text_ids, text_mask)
        B = text_ids.size(0)
        init = torch.full((B, 1), bos_id, dtype=torch.long, device=text_ids.device)
        return self.gpt2.generate(
            input_ids=init,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=text_mask,
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
