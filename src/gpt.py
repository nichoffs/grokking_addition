from tinygrad import Tensor, dtypes


class TransformerBlock:
    def __init__(self, embed_dim, head_dim, num_heads):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.q = Tensor.normal(embed_dim, embed_dim)
        self.k = Tensor.normal(embed_dim, embed_dim)
        self.v = Tensor.normal(embed_dim, embed_dim)

        self.head_out = Tensor.normal(num_heads * head_dim, embed_dim)

        self.ff1 = Tensor.normal(embed_dim, 4 * embed_dim)
        self.ff2 = Tensor.normal(4 * embed_dim, embed_dim)

    def attn(self, x):
        bsz = x.shape[0]
        q, k, v = [
            x.linear(proj)
            .reshape(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            for proj in (self.q, self.k, self.v)
        ]
        return (
            q.scaled_dot_product_attention(k, v)
            .transpose(1, 2)
            .reshape(bsz, -1, self.num_heads * self.head_dim)
            .linear(self.head_out)
        )

    def mlp(self, x):
        return x.linear(self.ff1).relu().linear(self.ff2)

    def __call__(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class GPT:
    def __init__(self, num_layers, embed_dim, vocab_size, context_length, num_heads):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_heads = num_heads

        self.tok_embed = Tensor.normal(vocab_size, embed_dim)
        self.pos_embed = Tensor.normal(context_length, embed_dim)

        self.blocks = [
            TransformerBlock(embed_dim, embed_dim // num_heads, num_heads)
            for _ in range(num_layers)
        ]

        self.out = Tensor.normal(embed_dim, vocab_size - 1)

    def __call__(self, x):
        # input shape (B,T,C)
        bsz = x.shape[0]
        pos = (
            Tensor.arange(self.context_length)
            .one_hot(self.context_length)
            .cast(dtypes.float)[: x.shape[1]]
            .expand((bsz, None, None))
        )
        x = x.one_hot(self.vocab_size).linear(self.tok_embed) + pos.linear(
            self.pos_embed
        )
        x = x.sequential(self.blocks)
        x = x.reshape(-1, x.shape[-1]).linear(self.out)
        return x.reshape((bsz, -1, x.shape[-1]))
