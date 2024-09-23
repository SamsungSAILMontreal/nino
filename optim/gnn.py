# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Based on https://github.com/mkofinas/neural-graphs/blob/main/nn/gnn.py

"""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Sequential
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear as pygLinear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import Adj, OptTensor


class PNA(torch.nn.Module):
    r"""A GNN model with modulated edge features from the Neural Graphs paper
    "Graph Neural Networks for Learning Equivariant Representations of Neural Networks" https://arxiv.org/abs/2403.12143
    based on "Principal Neighbourhood Aggregation for Graph Nets" https://arxiv.org/abs/2004.05718.

    Consists of a number of PNAConv and EdgeMLP layers (defined below).

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    supports_edge_weight = False
    supports_edge_attr = True

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        update_edge_attr: bool = False,
        final_edge_update: bool = True,
        chunk_size: int = 0,
        message_passing_device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs
        self.final_edge_update = final_edge_update
        self.chunk_size = chunk_size
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(self.init_conv(in_channels, hidden_channels,
                                             chunk_size=chunk_size, message_passing_device=message_passing_device,
                                             **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels

        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(self.init_conv(in_channels, out_channels,
                                             chunk_size=chunk_size, message_passing_device=message_passing_device,
                                             **kwargs))
        else:
            self.convs.append(self.init_conv(in_channels, hidden_channels,
                                             chunk_size=chunk_size, message_passing_device=message_passing_device,
                                             **kwargs))

        self.norms = None
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                hidden_channels,
                **(norm_kwargs or {}),
            )
            self.norms = ModuleList()
            for _ in range(num_layers):
                self.norms.append(copy.deepcopy(norm_layer))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None and jk != "last":
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == "cat":
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

        # Edge update stuff
        self.update_edge_attr = update_edge_attr
        if update_edge_attr:
            self.edge_update = nn.ModuleList(
                [
                    EdgeMLP(
                        edge_dim=kwargs["edge_dim"],
                        node_dim=hidden_channels,
                        act=self.act,
                        chunk_size=chunk_size,
                    )
                    for _ in range(num_layers if final_edge_update else num_layers - 1)
                ]
            )

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, "jk"):
            self.jk.reset_parameters()
        if hasattr(self, "lin"):
            self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj = None,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        if (
            num_sampled_nodes_per_hop is not None
            and isinstance(edge_weight, Tensor)
            and isinstance(edge_attr, Tensor)
        ):
            raise NotImplementedError(
                "'trim_to_layer' functionality does not "
                "yet support trimming of both "
                "'edge_weight' and 'edge_attr'"
            )

        xs: List[Tensor] = []
        for i in range(self.num_layers):
            if num_sampled_nodes_per_hop is not None:
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            upd_edge = self.update_edge_attr and (i < self.num_layers - 1 or self.final_edge_update)
            if not (i == self.num_layers - 1 and self.jk_mode is None) or upd_edge:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if hasattr(self, "jk"):
                    xs.append(x)

            # update edge representations
            if upd_edge:
                edge_attr = self.edge_update[i](x, edge_index, edge_attr)

                if self.norms is not None:
                    edge_attr = self.norms[i](edge_attr)

        x = self.jk(xs) if hasattr(self, "jk") else x
        x = self.lin(x) if hasattr(self, "lin") else x
        return x, edge_attr


class PNAConv(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left(
        \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
        h_{\mathbf{\Theta}} \left( \mathbf{x}_i, \mathbf{x}_j \right)
        \right)

    with

    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},

    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote MLPs.

    .. note::

        For an example of using :obj:`PNAConv`, see `examples/pna.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/
        examples/pna.py>`_.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (List[str]): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers (List[str]): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (torch.Tensor): Histogram of in-degrees of nodes in the training
            set, used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        act (str or callable, optional): Pre- and post-layer activation
            function to use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        train_norm (bool, optional): Whether normalization parameters
            are trainable. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Tensor = None,
        edge_dim: Optional[int] = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        train_norm: bool = False,
        modulate_edges: bool = False,
        gating_edges: bool = False,
        chunk_size=0,
        message_passing_device: Optional[torch.device] = None,
        **kwargs,
    ):
        if len(aggregators) == len(scalers) == 1 and scalers[0] == "identity":
            aggr = aggregators[0]
        else:
            aggr = DegreeScalerAggregation(aggregators, scalers, deg, train_norm)
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        divide_input = towers > 1
        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.modulate_edges = modulate_edges
        self.gating_edges = gating_edges
        self.chunk_size = chunk_size
        self.message_passing_device = message_passing_device

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        if self.edge_dim is not None:
            if modulate_edges:
                self.edge_encoder = pygLinear(edge_dim, 2 * self.F_in)
            else:
                self.edge_encoder = pygLinear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [
                pygLinear(
                    (3 if edge_dim and not modulate_edges else 2) * self.F_in, self.F_in
                )
            ]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [pygLinear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [pygLinear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [pygLinear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = pygLinear(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn_ in self.pre_nns:
            reset(nn_)
        for nn_ in self.post_nns:
            reset(nn_)
        self.lin.reset_parameters()

    def forward(
        self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        if self.message_passing_device is not None:
            device = x.device
            # perform the propagation on a separate device
            x = x.to(self.message_passing_device)
            edge_index = edge_index.to(self.message_passing_device)
            # if edge_attr is not None:
            #     edge_attr = edge_attr.to(self.message_passing_device)
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
            out = out.to(device)
            x = x.to(device)
        else:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)  # message passing and aggregation

        # node update (relatively cheap)
        out = torch.cat([x, out], dim=-1)
        out = torch.cat([nn(out[:, i]) for i, nn in enumerate(self.post_nns)], dim=1)
        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """
        This function is called in the self.propagate function of the forward pass above.
        """

        if edge_attr is not None:
            if self.modulate_edges:
                if self.training:
                    scale, shift = self.edge_encoder(edge_attr).chunk(2, dim=-1)
                    h = torch.cat([x_i, x_j], dim=-1)
            else:
                edge_attr = self.edge_encoder(edge_attr)
                edge_attr = edge_attr.view(-1, 1, self.F_in)
                edge_attr = edge_attr.repeat(1, self.towers, 1)
                h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        if self.modulate_edges and edge_attr is not None:
            if self.training:
                y = [scale * nn_(h[:, i]) + shift for i, nn_ in enumerate(self.pre_nns)]
            else:
                chunk_size = len(x_i) if self.chunk_size in [0, -1, None] else self.chunk_size
                assert len(self.pre_nns) == x_i.shape[1] == x_j.shape[1], (len(self.pre_nns),
                                                                           x_i.shape,
                                                                           x_j.shape)
                assert not torch.is_grad_enabled(), 'this must be run with torch.no_grad() to avoid memory leaks'

                for i, nn_ in enumerate(self.pre_nns):  # for each tower (by default 1)
                    device = nn_[0].weight.device if self.message_passing_device is not None else x_i.device
                    for j in range(0, len(x_i), chunk_size):  # chunking for memory efficiency
                        x_i[j:j + chunk_size, i] = nn_(torch.cat(
                            [x_i[j:j + chunk_size, i], x_j[j:j + chunk_size, i]],
                            dim=-1).to(device)).to(x_i)

                        scale, shift = self.edge_encoder(edge_attr[j:j + chunk_size]).chunk(2, dim=-1)

                        with torch.amp.autocast(enabled=False,
                                                device_type='cpu' if device == 'cpu' else 'cuda'):
                            # this operation is sensitive to precision, so we do it in float
                            x_i[j:j + chunk_size, i] = (scale.to(x_i).float() * x_i[j:j + chunk_size, i] +
                                                        shift.to(x_i).float()).to(x_i)
                y = x_i
        else:
            y = [nn_(h[:, i]) for i, nn_ in enumerate(self.pre_nns)]

        if isinstance(y, list):
            y = y[0].unsqueeze(1) if len(y) == 1 else torch.stack(y, dim=1)

        return y


class EdgeMLP(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        act: Callable,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.chunk_size = chunk_size
        self.lin_e = nn.Linear(edge_dim, edge_dim)
        self.lin_s = nn.Linear(node_dim, edge_dim)
        self.lin_t = nn.Linear(node_dim, edge_dim)
        self.act = act
        self.lin1 = nn.Linear(edge_dim, edge_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        if self.training:
            edge_attr = (
                self.lin_e(edge_attr)
                + self.lin_s(x)[edge_index[0]]
                + self.lin_t(x)[edge_index[1]]
            )
            edge_attr = self.act(edge_attr)
            edge_attr = self.lin1(edge_attr)
        else:
            assert not torch.is_grad_enabled(), 'this must be run with torch.no_grad() to avoid memory leaks'
            x_s = self.lin_s(x)
            x = self.lin_t(x)
            chunk_size = len(edge_attr) if self.chunk_size in [0, -1, None] else self.chunk_size
            # inplace operation with chunking, which is more memory efficient
            for i in range(0, len(edge_attr), chunk_size):
                edge_attr[i:i + chunk_size] = self.lin_e(edge_attr[i:i + chunk_size])
                edge_attr[i:i + chunk_size] += (x_s[edge_index[0][i:i + chunk_size]]
                                                + x[edge_index[1][i:i + chunk_size]])
                edge_attr[i:i + chunk_size] = self.act(edge_attr[i:i + chunk_size])
                edge_attr[i:i + chunk_size] = self.lin1(edge_attr[i:i + chunk_size])

        return edge_attr
