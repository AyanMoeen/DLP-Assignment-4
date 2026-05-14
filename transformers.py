"""
Implements a Transformer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import math

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F


def hello_transformers():
    print("Hello from transformers.py!")


def generate_token_dict(vocab):
    token_dict = {}
    for i, tok in enumerate(vocab):
        token_dict[tok] = i
    return token_dict


def prepocess_input_sequence(
    input_str: str, token_dict: dict, spc_tokens: list
) -> list:
    out = []
    for word in input_str.split():
        if word.isdigit():
            for ch in word:
                out.append(token_dict[ch])
        else:
            out.append(token_dict[word])
    return out


def scaled_dot_product_two_loop_single(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    K, M = query.shape
    out = query.new_zeros(K, M)
    scale = 1.0 / math.sqrt(M)
    for i in range(K):
        scores = query.new_zeros(K)
        for j in range(K):
            scores[j] = (query[i] * key[j]).sum()
        scores = scores * scale
        weights = F.softmax(scores, dim=0)
        out[i] = (weights.unsqueeze(1) * value).sum(dim=0)
    return out


def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    N, K, M = query.shape
    out = query.new_zeros(N, K, M)
    scale = 1.0 / math.sqrt(M)
    for n in range(N):
        for i in range(K):
            scores = (query[n, i:i+1] @ key[n].transpose(0, 1)).squeeze(0)
            scores = scores * scale
            weights = F.softmax(scores, dim=0)
            out[n, i] = (weights.unsqueeze(1) * value[n]).sum(dim=0)
    return out


def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
) -> Tensor:
    _, _, M = query.shape
    scale = 1.0 / math.sqrt(M)
    scores = torch.bmm(query, key.transpose(1, 2)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    weights_softmax = F.softmax(scores, dim=-1)
    y = torch.bmm(weights_softmax, value)
    return y, weights_softmax


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()
        self.weights_softmax = None

        c_q = math.sqrt(6 / (dim_in + dim_q))
        self.q = nn.Linear(dim_in, dim_q)
        nn.init.uniform_(self.q.weight, -c_q, c_q)
        nn.init.zeros_(self.q.bias)

        c_k = math.sqrt(6 / (dim_in + dim_q))
        self.k = nn.Linear(dim_in, dim_q)
        nn.init.uniform_(self.k.weight, -c_k, c_k)
        nn.init.zeros_(self.k.bias)

        c_v = math.sqrt(6 / (dim_in + dim_v))
        self.v = nn.Linear(dim_in, dim_v)
        nn.init.uniform_(self.v.weight, -c_v, c_v)
        nn.init.zeros_(self.v.bias)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:
        self.weights_softmax = None
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        y, self.weights_softmax = scaled_dot_product_no_loop_batch(q, k, v, mask)
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)]
        )
        concat_dim = num_heads * dim_out
        c = math.sqrt(6 / (concat_dim + dim_in))
        self.out_proj = nn.Linear(concat_dim, dim_in)
        nn.init.uniform_(self.out_proj.weight, -c, c)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:
        head_outs = [h(query, key, value, mask) for h in self.heads]
        concat = torch.cat(head_outs, dim=-1)
        y = self.out_proj(concat)
        return y


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std * self.gamma + self.beta
        return y


class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()
        c1 = math.sqrt(6 / (inp_dim + hidden_dim_feedforward))
        self.fc1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        nn.init.uniform_(self.fc1.weight, -c1, c1)
        nn.init.zeros_(self.fc1.bias)

        c2 = math.sqrt(6 / (hidden_dim_feedforward + inp_dim))
        self.fc2 = nn.Linear(hidden_dim_feedforward, inp_dim)
        nn.init.uniform_(self.fc2.weight, -c2, c2)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"emb_dim={emb_dim} not divisible by num_heads={num_heads}"
            )
        head_dim = emb_dim // num_heads
        self.mha      = MultiHeadAttention(num_heads, emb_dim, head_dim)
        self.norm1    = LayerNormalization(emb_dim)
        self.norm2    = LayerNormalization(emb_dim)
        self.ff       = FeedForwardBlock(emb_dim, feedforward_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        out1 = self.mha(x, x, x)
        out2 = self.dropout(self.norm1(out1 + x))
        out3 = self.ff(out2)
        y    = self.dropout(self.norm2(out3 + out2))
        return y


def get_subsequent_mask(seq):
    N, K = seq.shape
    mask = torch.triu(
        torch.ones((K, K), device=seq.device, dtype=torch.bool), diagonal=1
    )
    mask = mask.unsqueeze(0).expand(N, -1, -1)
    return mask


class DecoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"emb_dim={emb_dim} not divisible by num_heads={num_heads}"
            )
        head_dim = emb_dim // num_heads
        self.attention_self  = MultiHeadAttention(num_heads, emb_dim, head_dim)
        self.attention_cross = MultiHeadAttention(num_heads, emb_dim, head_dim)
        self.feed_forward    = FeedForwardBlock(emb_dim, feedforward_dim)
        self.norm1    = LayerNormalization(emb_dim)
        self.norm2    = LayerNormalization(emb_dim)
        self.norm3    = LayerNormalization(emb_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self, dec_inp: Tensor, enc_inp: Tensor, mask: Tensor = None
    ) -> Tensor:
        out1 = self.attention_self(dec_inp, dec_inp, dec_inp, mask)
        out2 = self.dropout(self.norm1(dec_inp + out1))
        out3 = self.attention_cross(out2, enc_inp, enc_inp)
        out4 = self.dropout(self.norm2(out2 + out3))
        out5 = self.feed_forward(out4)
        y    = self.dropout(self.norm3(out4 + out5))
        return y


class Encoder(nn.Module):
    def __init__(
        self, num_heads, emb_dim, feedforward_dim, num_layers, dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)
        return src_seq


class Decoder(nn.Module):
    def __init__(
        self, num_heads, emb_dim, feedforward_dim, num_layers, dropout, vocab_len
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
        a = (6 / (emb_dim + vocab_len)) ** 0.5
        nn.init.uniform_(self.proj_to_vocab.weight, -a, a)

    def forward(self, target_seq: Tensor, enc_out: Tensor, mask: Tensor):
        out = target_seq.clone()
        for _layer in self.layers:
            out = _layer(out, enc_out, mask)
        out = self.proj_to_vocab(out)
        return out


def position_encoding_simple(K: int, M: int) -> Tensor:
    pos = torch.arange(K, dtype=torch.float32).view(K, 1) / K
    y   = pos.expand(K, M).unsqueeze(0)
    return y


def position_encoding_sinusoid(K: int, M: int) -> Tensor:
    pe  = torch.zeros(K, M, dtype=torch.float32)
    pos = torch.arange(K, dtype=torch.float32).unsqueeze(1)
    pe[:, 0::2] = torch.sin(pos)
    pe[:, 1::2] = torch.cos(pos)
    y = pe.unsqueeze(0)
    return y


class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        dropout: float,
        num_enc_layers: int,
        num_dec_layers: int,
        vocab_len: int,
    ):
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_len, emb_dim)
        self.encoder   = Encoder(num_heads, emb_dim, feedforward_dim, num_enc_layers, dropout)
        self.decoder   = Decoder(num_heads, emb_dim, feedforward_dim, num_dec_layers, dropout, vocab_len)

    def forward(
        self, ques_b: Tensor, ques_pos: Tensor, ans_b: Tensor, ans_pos: Tensor
    ) -> Tensor:
        q_emb     = self.emb_layer(ques_b)
        a_emb     = self.emb_layer(ans_b)
        q_emb_inp = q_emb + ques_pos
        a_emb_inp = a_emb[:, :-1] + ans_pos[:, :-1]

        enc_out = self.encoder(q_emb_inp)
        mask    = get_subsequent_mask(ans_b[:, :-1])
        dec_out = self.decoder(a_emb_inp, enc_out, mask)
        dec_out = dec_out.reshape(-1, dec_out.size(-1))
        return dec_out


class AddSubDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_seqs,
        target_seqs,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        pos_encode,
    ):
        self.input_seqs            = input_seqs
        self.target_seqs           = target_seqs
        self.convert_str_to_tokens = convert_str_to_tokens
        self.emb_dim               = emb_dim
        self.special_tokens        = special_tokens
        self.pos_encode            = pos_encode

    def preprocess(self, inp):
        return prepocess_input_sequence(
            inp, self.convert_str_to_tokens, self.special_tokens
        )

    def __getitem__(self, idx):
        inp            = self.input_seqs[idx]
        out            = self.target_seqs[idx]
        preprocess_inp = torch.tensor(self.preprocess(inp))
        preprocess_out = torch.tensor(self.preprocess(out))
        inp_pos        = len(preprocess_inp)
        inp_pos_enc    = self.pos_encode(inp_pos, self.emb_dim)
        out_pos        = len(preprocess_out)
        out_pos_enc    = self.pos_encode(out_pos, self.emb_dim)
        return preprocess_inp, inp_pos_enc[0], preprocess_out, out_pos_enc[0]

    def __len__(self):
        return len(self.input_seqs)


def LabelSmoothingLoss(pred, ground):
    ground  = ground.contiguous().view(-1)
    eps     = 0.1
    n_class = pred.size(1)
    one_hot = torch.nn.functional.one_hot(ground).to(pred.dtype)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss    = -(one_hot * log_prb).sum(dim=1)
    return loss.sum()


def CrossEntropyLoss(pred, ground):
    return F.cross_entropy(pred, ground, reduction="sum")