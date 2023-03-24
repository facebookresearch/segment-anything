import math
from typing import Optional

import torch
from torch import Tensor, nn

class Transformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        p_dropout: float,
        activation: Optional[str] = "relu",
        pre_norm: Optional[bool] = False,
        add_pe_to_first_layer: bool = False,
        decoder_layer: Optional[nn.Module] = DecoderLayer,
        final_attention_by_clicks: bool = False,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth: number of layers in the transformer
          embedding_dim: the channel dimension for the input embeddings
          num_heads: the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim: the channel dimension internal to the MLP block
          p_dropout: the dropout probability. Applied after each block,
            in the MLP block, and as attention dropout in both attention
            layers.
          activation: the activation to use in the MLP block. Must be
            'relu' or 'gelu'
          pre_norm: If True, layer norm is applied at the beginning of
            the residual of each block. If False, layer norm is applied
            after summing the residual and identity branch of each block.
          decorder_layer: The type of decoder layer to use. Defaults to DecoderLayer
          final_attention_by_clicks: If true, applies one final downsample=2 attention
            layer of the clicks to the image. Exists mostly to assure useful information
            is propagated from the image embedding to the clicks for TFO models when
            we are trying to get an IoU prediction from the click output token.
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.p_dropout = p_dropout
        self.pre_norm = pre_norm
        self.layers = nn.ModuleList()

        for _i in range(depth):
            self.layers.append(
                decoder_layer(
                    embedding_dim,
                    num_heads,
                    mlp_dim,
                    p_dropout,
                    activation,
                    pre_norm,
                    add_pe_to_first_layer,
                    attention_downsample_rate,
                )
            )
        # FIXME: When pre_norm = False, layer norm is applied twice in
        # a row to the final output. Maintaining this behavior at the
        # moment for reproducability.
        if decoder_layer is DecoderLayer:
            self.final_norm = nn.LayerNorm(embedding_dim)
        else:
            self.final_norm = None

        if final_attention_by_clicks:
            self.final_attn = Attention(
                embedding_dim, num_heads, p_dropout, downsample_rate=attention_downsample_rate
            )
            self.norm_final_attn = nn.LayerNorm(embedding_dim)
            self.dropout_final_attn = nn.Dropout(p_dropout)
        else:
            self.final_attn = None
            self.norm_final_attn = None
            self.dropout_final_attn = None

        self._reset_parameters()

    def _reset_parameters(self):
        # FIXME: This is the same code as in the old version but gives
        # a slightly different result since there are 3 CxC weights
        # in the attention block instead of one 3CxC weight.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tensor:
        """
        Args:
          image_embedding: image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe: the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding: the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Output:
          Returns a tensor of the same shape as point_embedding.
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = torch.zeros_like(point_embedding)
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for i, layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                i_layer=i,
            )

        if self.final_attn is not None:
            queries_res = self.norm_final_attn(queries) if self.pre_norm else queries
            q = queries_res + point_embedding
            k = keys + image_pe
            attn_out = self.final_attn(q=q, k=k, v=keys)
            queries = queries + self.dropout_final_attn(attn_out)
            queries = self.norm_final_attn(queries) if not self.pre_norm else queries

        # Apply final layer norm
        if self.final_norm is not None:
            queries = self.final_norm(queries)

        return queries, keys


class TwoWayDecoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        p_dropout: float = 0.1,
        activation: str = "relu",
        pre_norm: bool = False,
        add_pe_to_first_layer: bool = False,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder layer. Applies in order: self attention
        on the queries, cross attention between queries and keys, and
        an MLP layer on the queries.

        Args:
          embedding_dim: the channel dimension of queries and keys
          num_heads: the number of heads for multihead attention. Must
            divide embedding_dim.
          mlp_dim: the internal dimension of the mlp block
          p_dropout: the dropout probability. Applied after each block,
            in the MLP block, and as attention dropout in both attention
            layers.
          activation: the activation to use in the MLP block. Must be
            'relu' or 'gelu'
          pre_norm: If True, layer norm is applied at the beginning of
            the residual of each block. If False, layer norm is applied
            after summing the residual and identity branch of each block.
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads, p_dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(p_dropout)

        self.cross_attn_image = Attention(
            embedding_dim, num_heads, p_dropout, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(p_dropout)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation, p_dropout)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout3 = nn.Dropout(p_dropout)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.dropout4 = nn.Dropout(p_dropout)
        self.cross_attn_clicks = Attention(
            embedding_dim, num_heads, p_dropout, downsample_rate=attention_downsample_rate
        )

        self.pre_norm = pre_norm

        self.add_pe_to_first_layer = add_pe_to_first_layer

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, i_layer: int
    ) -> Tensor:
        # Self attention block
        queries_res = self.norm1(queries) if self.pre_norm else queries
        q = queries_res + query_pe
        if self.add_pe_to_first_layer:
            # prevent v being set as zero tensor in the first layer
            attn_out = (
                self.self_attn(q=q, k=q, v=queries_res)
                if i_layer > 0
                else self.self_attn(q=q, k=q, v=q)
            )
        else:
            attn_out = self.self_attn(q=q, k=q, v=queries_res)
        queries = queries + self.dropout1(attn_out)
        queries = self.norm1(queries) if not self.pre_norm else queries

        # Cross attention block, clicks attending to image features
        queries_res = self.norm2(queries) if self.pre_norm else queries
        q = queries_res + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image(q=q, k=k, v=keys)
        queries = queries + self.dropout2(attn_out)
        queries = self.norm2(queries) if not self.pre_norm else queries

        # MLP block
        mlp_in = self.norm3(queries) if self.pre_norm else queries
        mlp_out = self.mlp(mlp_in)
        queries = queries + self.dropout3(mlp_out)
        queries = self.norm3(queries) if not self.pre_norm else queries

        # Cross attention block, image features attending to clicks
        keys_res = self.norm4(keys) if self.pre_norm else keys
        q = queries + query_pe
        k = keys_res + key_pe
        attn_out = self.cross_attn_clicks(q=k, k=q, v=queries)
        keys = keys + self.dropout4(attn_out)
        keys = self.norm4(keys) if not self.pre_norm else keys

        return queries, keys


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        downsample_rate: int = 1,
    ) -> None:
        """
        Simplified attention. It is batch first, doesn't mask, and assumes
        equal embedding dimensions for qkv. Expects input as BxNxC for N tokens.

        It processes qkv linear layers serially instead of as a single matrix,
        which may be slightly slower than nn.MultiheadAttention but allows for
        easier customization, and avoids various torch script problems.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.q_dropout = nn.Dropout(proj_dropout)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_dropout = nn.Dropout(proj_dropout)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_dropout = nn.Dropout(proj_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.out_dropout = nn.Dropout(proj_dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # The fan_out is incorrect, but matches pytorch's initialization
        # for which qkv is a single 3*embedding_dim x embedding_dim matrix
        fan_in = self.embedding_dim
        fan_out = 3 * self.internal_dim
        # Xavier uniform with our custom fan_out
        bnd = math.sqrt(6 / (fan_in + fan_out))
        nn.init.uniform_(self.q_proj.weight, -bnd, bnd)
        nn.init.uniform_(self.k_proj.weight, -bnd, bnd)
        nn.init.uniform_(self.v_proj.weight, -bnd, bnd)

        # out_proj.weight is left with default initialization, like pytorch attention

        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_dropout(self.q_proj(q))
        k = self.k_dropout(self.k_proj(k))
        v = self.v_dropout(self.v_proj(v))

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_dropout(self.out_proj(out))

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        activation: str,
        p_dropout: float,
    ) -> None:
        """
        A simple MLP block consisting of two linear layers, an activation
        and dropout.
        """

        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = self._get_act(activation)
        self.dropout = nn.Dropout(p_dropout)

    def _get_act(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise NotImplementedError(f"activation should be relu/gelu, not {activation}.")

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.dropout(self.act(self.lin1(x))))
