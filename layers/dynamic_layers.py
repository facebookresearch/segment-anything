# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def prod(iterable):
    out = 1
    for elem in iterable:
        out *= elem
    return out


class DynamicMLP(nn.Module):
    """
    Applies a multilayer hypernetwork to a batch of input images.

    Expects input features of shape (..., in_dim, H, W) and input weights
    in the form (..., N_weights). Flattens H x W, then 'features' is
    multiplied by weights on the left, so this network maps the shapes
    (B, in_dim, HW) -> (..., out_dim, HW). Weights are provided as a
    single dimension (..., num_weights) aand split by the hypernetwork
    into various components. The ... part must be broadcastable between
    'features' and 'weights'. Reshapes output to (..., out_dim, H, W)

    Can calculate and return an L2 loss associated with the dynamic
    weight.

    Args:
      in_dim: the input dimension of the MLP
      hidden_dim: the dimension of internal layers of the MLP
      out_dim: the output dimension of the mlp
      depth: the number of layers of the MLP. If depth=1, the MLP is
        just a linear layer so 'hidden_dim' and 'activation' don't matter.
      regularization: scale factor for L2 loss.
      activation: the activation to use. Supports 'relu' and 'gelu'.
      bias: if true, applies bias in the MLP
      regularizer_mode: If 'mean', averages L2 loss over weights, if
        'sum', sums over weights.
      regularizer_bias: If true, L2 regularization is applied to bias
        terms as well.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        regularizer: int,
        activation: str = "relu",
        bias: bool = True,
        regularizer_mode: str = "mean",
        regularizer_bias: bool = True,
        max_eval_parallelism: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.has_bias = bias
        self.regularizer_mode = regularizer_mode
        self.regularizer_bias = regularizer_bias
        self.regularizer = regularizer
        self.max_eval_parallelism = max_eval_parallelism
        assert regularizer_mode in ["mean", "sum"]
        if depth == 1:
            self.act = None
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise NotImplementedError(
                f"Activation should be in ['relu', 'gelu'], got {activation}."
            )
        self.weight_shapes = self._build_weight_shapes(in_dim, hidden_dim, out_dim, depth)
        self.num_weights = sum(prod(i) for i in self.weight_shapes.values())

    def _build_weight_shapes(
        self, in_dim: int, hidden_dim: int, out_dim: int, layers: int
    ) -> Dict[str, Tuple[int]]:
        weight_shapes = {}
        act_shapes = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        for i in range(layers):
            layer_in_dim = act_shapes[i]
            layer_out_dim = act_shapes[i + 1]
            weight_shapes[f"layer_{i+1}_weight"] = (layer_out_dim, layer_in_dim)
            if self.has_bias:
                weight_shapes[f"layer_{i+1}_bias"] = (layer_out_dim, 1)
        return weight_shapes

    def _split_weights(self, weights: Tensor) -> Dict[str, Tensor]:
        weight_dict = {}
        idx = 0
        w_shape = weights.shape
        for k, v in self.weight_shapes.items():
            num_weights = prod(v)
            out_shape = (*w_shape[:-1], *v)
            weight_dict[k] = weights[..., idx : idx + num_weights].view(*out_shape)
            idx += num_weights
        return weight_dict

    def _apply_mlp(self, x: Tensor, weight_dict: Dict[str, Tensor]) -> Tensor:
        for i in range(self.depth):
            w = weight_dict[f"layer_{i+1}_weight"]
            x = w @ x
            if self.has_bias:
                b = weight_dict[f"layer_{i+1}_bias"]
                x = x + b
            if i < self.depth - 1:
                x = self.act(x)
        return x

    def _chunk_weight_dict(
        self, weight_dict: Dict[str, Tensor], size: int
    ) -> List[Dict[str, Tensor]]:
        chunks = None
        for k, v in weight_dict.items():
            shape = v.shape
            flat_v = v.reshape(-1, shape[-2], shape[-1])
            n_chunks = -(flat_v.shape[0] // -size)  # Ceiling division
            if chunks is None:
                chunks = [{} for i in range(n_chunks)]
            for n in range(n_chunks):
                chunks[n][k] = flat_v[n * size : (n + 1) * size]
        return chunks

    def _apply_mlp_by_chunk(self, x: Tensor, weight_dict: Dict[str, Tensor], size: int) -> Tensor:
        weight_dicts = self._chunk_weight_dict(weight_dict, size)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])
        out = torch.empty(x.shape[0], self.out_dim, x.shape[-1]).to(x.device)
        n_chunks = _ceildiv(x.shape[0], size)
        for n in range(n_chunks):
            out[n * size : (n + 1) * size] = self._apply_mlp(
                x[n * size : (n + 1) * size], weight_dicts[n]
            )
        out = out.reshape(*x_shape[:-2], self.out_dim, x_shape[-1])
        return out

    def forward(self, features: Tensor, weights: Tensor, force_no_chunk: bool = False) -> Tensor:
        weight_dict = self._split_weights(weights)
        in_shape = features.shape
        # if delete_input:
        #    x = deepcopy(features.flatten(-2))
        #    del features
        # else:
        #    x = features.flatten(-2)
        x = features.flatten(-2)

        num_features = prod(in_shape[:-2])
        n_chunks = (
            _ceildiv(num_features, self.max_eval_parallelism)
            if self.max_eval_parallelism is not None and not force_no_chunk
            else 1
        )
        if n_chunks == 1 or self.training:
            x = self._apply_mlp(x, weight_dict)
        else:
            x = self._apply_mlp_by_chunk(x, weight_dict, self.max_eval_parallelism)

        out_shape = in_shape[:-3] + (self.out_dim,) + in_shape[-2:]
        return x.reshape(out_shape)

    def get_loss(self, weights: Tensor) -> Tensor:
        if self.regularizer == 0.0:
            return torch.zeros(1, device=weights.device)
        if self.regularizer_bias or not self.has_bias:
            if self.regularizer_mode == "mean":
                return self.regularizer * (weights ** 2).mean()
            else:
                return self.regularizer * (weights ** 2).sum()

        weight_dict = self._split_weights(weights)
        loss = 0.0
        count = 0
        for i in range(self.depth):
            loss = loss + self.regularizer * (weight_dict[f"layer_{i+1}_weight"] ** 2).sum()
            count = count + prod(self.weight_shapes[f"layer_{i+1}_weight"])
        if self.regularizer_mode == "mean":
            loss = loss / count
        return loss


class DynamicLinear(DynamicMLP):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(in_dim, -1, out_dim, depth=1, regularizer=0.0, bias=False)

def _ceildiv(a: int, b: int) -> int:
    return -(a // -b)
