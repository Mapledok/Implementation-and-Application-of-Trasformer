"""
@Time : 2023/4/7 14:16
@Author : 十三
@Email : mapledok@outlook.com
@File : FullModel.py
@Project : Transformer
"""
import copy
import torch
import torch.nn as nn
from basic import show_example
from MutiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward, PositionalEncoding, Embeddings
from ModelArchitecture import EncoderDecoder, Generator
from EncoderDecoderStacks import Encoder, EncoderLayer, Decoder, DecoderLayer, subsequent_mask


# Here we define a function from hyperparameters to a full model.
def make_model(
        src_vocab, tgt_vocab, N=6, d_model=512, d_ffn=2048, num_heads=8, dropout=0.1
):
    """
    Helper: Construct a model from hyperparameters.
    """

    c = copy.deepcopy
    attn = MultiHeadAttention(num_heads, d_model)
    ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoder(
            layer=EncoderLayer(
                size=d_model,
                self_attn=c(attn),
                feed_forward=c(ffn),
                dropout=dropout),
            N=N
        ),
        decoder=Decoder(
            layer=DecoderLayer(
                size=d_model,
                self_attn=c(attn),
                src_attn=c(attn),
                feed_forward=c(ffn),
                dropout=dropout),
            N=N
        ),
        src_embed=nn.Sequential(
            Embeddings(d_model=d_model, vocab=src_vocab),
            c(position)
        ),
        tgt_embed=nn.Sequential(
            Embeddings(d_model=d_model, vocab=tgt_vocab),
            c(position)
        ),
        generator=Generator(d_model=d_model, vocab=tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    def weights_init(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_normal_(m.weight)  # Gaussian or Random distribution

    model.apply(weights_init)

    return model


def inference_test():
    test_model = make_model(src_vocab=11, tgt_vocab=11, N=2)
    test_model.eval()
    src = torch.LongTensor([list(range(1, 11))])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)],
            dim=1
        )

    print('Example Untrained Model Prediction:', ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)
