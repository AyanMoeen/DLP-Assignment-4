import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction


def hello_rnn_lstm_captioning():
    print("Hello from rnn_lstm_captioning.py!")


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, verbose: bool = True):
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]
        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, images: torch.Tensor):
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0
        images = self.normalize(images)
        features = self.backbone(images)["c5"]
        return features


##############################################################################
# Recurrent Neural Network                                                   #
##############################################################################
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store next
    # hidden state and any values you need for the backward pass in the next_h
    # and cache variables respectively.
    ##########################################################################
    next_h = torch.tanh(x @ Wx + prev_h @ Wh + b)
    cache = (x, prev_h, Wx, Wh, b, next_h)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.
    ##########################################################################
    x, prev_h, Wx, Wh, b, next_h = cache
    dtanh = (1 - next_h ** 2) * dnext_h
    dx = dtanh @ Wx.T
    dprev_h = dtanh @ Wh.T
    dWx = x.T @ dtanh
    dWh = prev_h.T @ dtanh
    db = dtanh.sum(axis=0)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = None, None
    ##########################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of
    # input data.
    ##########################################################################
    N, T, D = x.shape
    _, H = h0.shape
    h = torch.zeros(N, T, H, dtype=x.dtype, device=x.device)
    cache = []
    prev_h = h0
    for t in range(T):
        prev_h, c = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = prev_h
        cache.append(c)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return h, cache


def rnn_backward(dh, cache):
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire
    # sequence of data.
    ##########################################################################
    N, T, H = dh.shape
    _, D = cache[0][0].shape
    dx = torch.zeros(N, T, D, dtype=dh.dtype, device=dh.device)
    dWx = torch.zeros_like(cache[0][2])
    dWh = torch.zeros_like(cache[0][3])
    db = torch.zeros_like(cache[0][4])
    dprev_h = torch.zeros(N, H, dtype=dh.dtype, device=dh.device)
    for t in reversed(range(T)):
        dnext_h = dh[:, t, :] + dprev_h
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    dh0 = dprev_h
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):
        out = None
        ######################################################################
        # TODO: Implement the forward pass for word embeddings.
        ######################################################################
        out = self.W_embed[x]
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    loss = None
    ##########################################################################
    # TODO: Implement the temporal softmax loss function.
    # REQUIREMENT: This part MUST be done in one single line of code!
    ##########################################################################
    loss = F.cross_entropy(x.reshape(-1, x.shape[2]), y.reshape(-1), ignore_index=ignore_index, reduction='sum') / x.shape[0]
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return loss


class CaptioningRNN(nn.Module):
    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = "rnn",
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index

        ######################################################################
        # TODO: Initialize the image captioning module.
        ######################################################################
        self.image_encoder = ImageEncoder(pretrained=image_encoder_pretrained)
        self.embedding = WordEmbedding(vocab_size, wordvec_dim)
        if cell_type == 'rnn':
            self.rnn = RNN(wordvec_dim, hidden_dim)
        elif cell_type == 'lstm':
            self.rnn = LSTM(wordvec_dim, hidden_dim)
        elif cell_type == 'attn':
            self.rnn = AttentionLSTM(wordvec_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def forward(self, images, captions):
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        ######################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.
        ######################################################################
        features = self.image_encoder(images)
        if self.cell_type == 'attn':
            A = self.feature_proj(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            word_embed = self.embedding(captions_in)
            hidden = self.rnn(word_embed, A)
        else:
            pooled = features.mean(dim=(2, 3))
            h0 = self.feature_proj(pooled)
            word_embed = self.embedding(captions_in)
            hidden = self.rnn(word_embed, h0)
        scores = self.output_proj(hidden)
        loss = temporal_softmax_loss(scores, captions_out, ignore_index=self.ignore_index)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return loss

    def sample(self, images, max_length=15):
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        ######################################################################
        # TODO: Implement test-time sampling for the model.
        ######################################################################
        features = self.image_encoder(images)

        if self.cell_type == 'attn':
            A = self.feature_proj(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            prev_h = A.mean(dim=(2, 3))
            prev_c = prev_h.clone()
        else:
            pooled = features.mean(dim=(2, 3))
            prev_h = self.feature_proj(pooled)
            prev_c = torch.zeros_like(prev_h)

        word = torch.full((N,), self._start, dtype=torch.long, device=prev_h.device)

        for t in range(max_length):
            embed = self.embedding(word.unsqueeze(1)).squeeze(1)
            if self.cell_type == 'rnn':
                prev_h = self.rnn.step_forward(embed, prev_h)
            elif self.cell_type == 'lstm':
                prev_h, prev_c = self.rnn.step_forward(embed, prev_h, prev_c)
            elif self.cell_type == 'attn':
                attn, attn_weights = dot_product_attention(prev_h, A)
                prev_h, prev_c = self.rnn.step_forward(embed, prev_h, prev_c, attn)
                attn_weights_all[:, t, :, :] = attn_weights
            scores = self.output_proj(prev_h)
            word = scores.argmax(dim=1)
            captions[:, t] = word
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ######################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.
        ######################################################################
        next_h, next_c = None, None
        H = prev_h.shape[1]
        A = x @ self.Wx + prev_h @ self.Wh + self.b
        i, f, o, g = A.split(H, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        c0 = torch.zeros_like(h0)
        ######################################################################
        # TODO: Implement the forward pass for an LSTM over entire timeseries
        ######################################################################
        hn = None
        N, T, D = x.shape
        H = h0.shape[1]
        hn = torch.zeros(N, T, H, dtype=x.dtype, device=x.device)
        prev_h = h0
        prev_c = c0
        for t in range(T):
            prev_h, prev_c = self.step_forward(x[:, t, :], prev_h, prev_c)
            hn[:, t, :] = prev_h
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return hn


def dot_product_attention(prev_h, A):
    N, H, D_a, _ = A.shape
    attn, attn_weights = None, None
    ##########################################################################
    # TODO: Implement the scaled dot-product attention.
    ##########################################################################
    A_flat = A.reshape(N, H, -1)                                            # (N, H, 16)
    scores = torch.bmm(prev_h.unsqueeze(1), A_flat)                        # (N, 1, 16)
    scores = scores / (H ** 0.5)
    attn_weights = F.softmax(scores, dim=2)                                 # (N, 1, 16)
    attn = torch.bmm(attn_weights, A_flat.permute(0, 2, 1)).squeeze(1)     # (N, H)
    attn_weights = attn_weights.squeeze(1).reshape(N, D_a, D_a)            # (N, 4, 4)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return attn, attn_weights


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #######################################################################
        # TODO: Implement forward pass for a single timestep of attention LSTM.
        #######################################################################
        next_h, next_c = None, None
        H = prev_h.shape[1]
        A = x @ self.Wx + prev_h @ self.Wh + attn @ self.Wattn + self.b
        i, f, o, g = A.split(H, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        h0 = A.mean(dim=(2, 3))
        c0 = h0

        ######################################################################
        # TODO: Implement the forward pass for an LSTM over an entire time-
        # series using dot_product_attention.
        ######################################################################
        hn = None
        N, T, D = x.shape
        H = h0.shape[1]
        hn = torch.zeros(N, T, H, dtype=x.dtype, device=x.device)
        prev_h = h0
        prev_c = c0
        for t in range(T):
            attn, _ = dot_product_attention(prev_h, A)
            prev_h, prev_c = self.step_forward(x[:, t, :], prev_h, prev_c, attn)
            hn[:, t, :] = prev_h
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return hn