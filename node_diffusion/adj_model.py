"""
Autoregressive model: Chinese prompt -> adjacency matrix token sequence.

Encoder : frozen BERT (bert-base-chinese)  -> [B, src_len, 768]
Decoder : Transformer decoder (cross-attn) -> [B, 98, 256] logits
Loss    : CrossEntropy
"""

import torch
import torch.nn as nn
from transformers import BertModel


SEQ_LEN   = 98    # number of 8-bit tokens
VOCAB_SIZE = 256  # 0~255


class AdjDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # token embedding for target sequence (0~255 + BOS)
        self.token_emb = nn.Embedding(VOCAB_SIZE + 1, d_model)  # +1 for BOS=256
        self.pos_emb   = nn.Embedding(SEQ_LEN + 1, d_model)     # +1 for BOS position

        # project BERT 768 -> d_model for cross-attention
        self.enc_proj = nn.Linear(768, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, enc_out, enc_mask, tgt_tokens):
        """
        enc_out    : [B, src_len, 768]  BERT output
        enc_mask   : [B, src_len]       1=padding (to be ignored)
        tgt_tokens : [B, seq_len]       target token ids (0~255)

        Returns logits [B, seq_len, 256]
        """
        B, seq_len = tgt_tokens.shape

        # prepend BOS=256, drop last token (teacher forcing)
        bos  = torch.full((B, 1), VOCAB_SIZE, dtype=torch.long, device=tgt_tokens.device)
        tgt  = torch.cat([bos, tgt_tokens[:, :-1]], dim=1)    # [B, seq_len]

        pos  = torch.arange(seq_len, device=tgt.device).unsqueeze(0)
        emb  = self.token_emb(tgt) + self.pos_emb(pos)        # [B, seq_len, d_model]

        # causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tgt.device
        )

        mem  = self.enc_proj(enc_out)                          # [B, src_len, d_model]

        out  = self.decoder(
            tgt=emb,
            memory=mem,
            tgt_mask=causal_mask,
            memory_key_padding_mask=enc_mask.bool(),
        )                                                      # [B, seq_len, d_model]

        return self.out_proj(out)                              # [B, seq_len, 256]

    @torch.no_grad()
    def generate(self, enc_out, enc_mask):
        """Greedy autoregressive generation. Returns tokens [B, 98]."""
        B = enc_out.size(0)
        mem = self.enc_proj(enc_out)

        generated = torch.full((B, 1), VOCAB_SIZE, dtype=torch.long, device=enc_out.device)

        for i in range(SEQ_LEN):
            pos  = torch.arange(generated.size(1), device=enc_out.device).unsqueeze(0)
            emb  = self.token_emb(generated) + self.pos_emb(pos)
            mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1), device=enc_out.device
            )
            out   = self.decoder(emb, mem, tgt_mask=mask,
                                 memory_key_padding_mask=enc_mask.bool())
            logit = self.out_proj(out[:, -1, :])               # [B, 256]
            next_token = logit.argmax(dim=-1, keepdim=True)    # [B, 1]
            generated  = torch.cat([generated, next_token], dim=1)

        return generated[:, 1:]                                # [B, 98], remove BOS


class AdjGenerationModel(nn.Module):
    def __init__(self, bert_path='models/bert-base-chinese',
                 d_model=256, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_path)
        for p in self.bert.parameters():
            p.requires_grad = False

        self.decoder = AdjDecoder(d_model, nhead, num_layers, dropout)
        self.loss_fn = nn.CrossEntropyLoss()

        n_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"AdjGenerationModel decoder: {n_params:,} parameters (BERT frozen)")

    def forward(self, input_ids, attention_mask, tgt_tokens):
        """
        input_ids      : [B, src_len]  BERT input
        attention_mask : [B, src_len]  1=valid, 0=padding
        tgt_tokens     : [B, 98]       target adj tokens

        Returns scalar loss.
        """
        with torch.no_grad():
            enc_out = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state

        enc_mask = (1 - attention_mask)                        # 1=padding for decoder
        logits   = self.decoder(enc_out, enc_mask, tgt_tokens) # [B, 98, 256]

        loss = self.loss_fn(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_tokens.reshape(-1).long()
        )
        return loss

    @torch.no_grad()
    def generate(self, input_ids, attention_mask):
        enc_out  = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
        enc_mask = (1 - attention_mask)
        return self.decoder.generate(enc_out, enc_mask)        # [B, 98]
