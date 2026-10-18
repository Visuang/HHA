import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math
from collections import OrderedDict


class HypLinear(nn.Module):
    """
    Poincare linear layer.
    """
    def __init__(self, manifold, in_features, out_features, c, dropout=0.6, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HyperbolicLayerNorm(nn.Module):
    def __init__(self, embedding_dim, manifold, c, eps=1e-5):
        super(HyperbolicLayerNorm, self).__init__()
        self.manifold = manifold
        self.c = c
        self.eps = eps


        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):


        x_tangent = self.to_tangent(x, c=self.c)


        x_norm = self.norm(x_tangent)


        return self.to_hyper(x_norm, c=self.c)


    def to_hyper(self, x, c=1.0):
        if self.manifold.name == 'PoincareBall':
            x_tan = self.manifold.proj_tan0(x, c)
            x_hyp = self.manifold.expmap0(x_tan, c)
            x_hyp = self.manifold.proj(x_hyp, c)
        elif self.manifold.name == 'Lorentz':
            x_tan = self.manifold.proju0(x, c)
            x_hyp = self.manifold.expmap0(x_tan, c)
        return x_hyp

    def to_tangent(self, x, c=1.0):
        if self.manifold.name == 'PoincareBall':
            x_tan = self.manifold.logmap0(x, c)
            x_tan = self.manifold.proj_tan0(x_tan, c)
        elif self.manifold.name == 'Lorentz':
            x_tan = self.manifold.log_map_zero(x, c)
            x_tan = self.manifold.normalize_tangent_zero(x_tan, c)
        return x_tan


class HyperbolicQuickGELU(nn.Module):
    def __init__(self, manifold, c):
        super(HyperbolicQuickGELU, self).__init__()
        self.manifold = manifold
        self.c = c
    def forward(self, x: torch.Tensor):
        x_tan = self.manifold.logmap0(x, self.c)

        x_tan = x_tan * torch.sigmoid(1.702 * x_tan)
        return self.manifold.expmap0(x_tan, self.c)


class HyperbolicMultiheadAttention2(nn.Module):
    def __init__(self, embedding_dim, num_heads, manifold, c, dropout, dropconnect_prob=0.1):
        super(HyperbolicMultiheadAttention2, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.c = c
        self.head_dim = embedding_dim // num_heads

        self.query_proj = HypLinear(self.manifold, embedding_dim, embedding_dim, c=self.c, dropout=0., use_bias=1)
        self.key_proj = HypLinear(self.manifold, embedding_dim, embedding_dim, c=self.c, dropout=0., use_bias=1)
        self.value_proj = HypLinear(self.manifold, embedding_dim, embedding_dim, c=self.c, dropout=0., use_bias=1)

        self.out_proj = HypLinear(self.manifold, embedding_dim, embedding_dim, c=self.c, dropout=0., use_bias=1)

        self.head_scaling = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dropconnect_prob = dropconnect_prob

    def forward(self, q, k, v):
        q_batch_size, q_seq_length, q_embed_dim = q.size()
        k_batch_size, k_seq_length, k_embed_dim = k.size()
        v_batch_size, v_seq_length, v_embed_dim = v.size()

        q = self.query_proj(q).view(q_batch_size, q_seq_length, self.num_heads, self.head_dim)
        k = self.key_proj(k).view(k_batch_size, k_seq_length, self.num_heads, self.head_dim)
        v = self.value_proj(v).view(v_batch_size, v_seq_length, self.num_heads, self.head_dim)

        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]


        q_tan = self.manifold.logmap0(q, self.c)


        k_tan = self.manifold.logmap0(k, self.c)


        v_tan = self.manifold.logmap0(v, self.c)


        attn_scores = torch.matmul(q_tan, k_tan.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores / self.head_scaling


        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.training and self.dropconnect_prob > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropconnect_prob, training=True)


        attn_output_tangent = torch.matmul(attn_weights, v_tan)


        attn_output = attn_output_tangent.transpose(1, 2).reshape(v_batch_size, -1, self.num_heads * self.head_dim)


        attn_output = self.manifold.expmap0(attn_output, self.c)
        output = self.out_proj(attn_output)

        return output, attn_output


class HyperbolicResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout, manifold, c, attn_mask: torch.Tensor = None):
        super().__init__()

        self.manifold = manifold
        self.c = c
        self.hyp_attn = HyperbolicMultiheadAttention2(d_model, n_head, manifold, c=self.c, dropout=dropout)
        self.hyp_ln1 = HyperbolicLayerNorm(d_model, manifold, c=self.c)
        self.hyp_mlp = nn.Sequential(OrderedDict([
            ("c_fc", HypLinear(self.manifold, d_model, d_model * 4, c=self.c, dropout=dropout, use_bias=1)),
            ("gelu", HyperbolicQuickGELU(self.manifold, c=self.c)),
            ("c_proj", HypLinear(self.manifold, d_model * 4, d_model, c=self.c, dropout=dropout, use_bias=1))
        ]))
        self.hyp_ln2 = HyperbolicLayerNorm(d_model, manifold, c=self.c)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x1 = self.hyp_ln1(x)
        x1, _ = self.hyp_attn(x1, x1, x1)
        x = self.manifold.mobius_add(x, x1, c=self.c)
        x = self.manifold.mobius_add(x, self.hyp_mlp(self.hyp_ln2(x)), c=self.c)
        return x


class HyperbolicTransformer(nn.Module):
    def __init__(self, layers: int, d_model: int, heads: int, dropout, manifold, c, attn_mask: torch.Tensor = None):
        super().__init__()
        self.d_model = d_model
        self.layers = layers
        self.resblocks = nn.Sequential(*[HyperbolicResidualAttentionBlock(d_model, heads, dropout, manifold, c, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
