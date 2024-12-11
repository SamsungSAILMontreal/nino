# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn.resolver import activation_resolver
from .gnn import PNA


class NiNoModel(nn.Module):

    """
    NiNo model for predicting future parameters.
    Default arguments are set for our best performing NiNo model.
    """

    def __init__(self,
                 ctx=5,
                 hid=128,
                 layers=3,
                 gnn=True,
                 dms=True,
                 max_feat_size=9,  # assuming max 3x3 conv
                 input_layer='linear',
                 act_name='silu',
                 residual=True,
                 seq_len=40,
                 improved_graph=True,
                 wte_pos_enc=True,  # ignored for mlp
                 vocab_size=50257,  # 50257 for GPT2, ignored for mlp
                 edge_types=15,
                 lpe=8,  # ignored for mlp
                 chunk_size=10**6,
                 message_passing_device=None,
                 scale_method='std',
                 upd_scale=1.0,
                 edge_sample_ratio=0.0,
                 **kwargs):
        super().__init__()

        self.ctx = ctx
        self.hid = hid
        self.max_feat_size = max_feat_size
        self.dms = dms
        if not self.dms:
            raise NotImplementedError('Only dms=True is supported in this implementation')
        self.residual = residual
        self.seq_len = seq_len
        self.max_feat_size = 1 if max_feat_size is None else max_feat_size
        self.improved_graph = improved_graph
        if not self.improved_graph:
            raise NotImplementedError('Only improved_graph=True is supported in this implementation')
        self.wte_pos_enc = wte_pos_enc
        self.edge_types = edge_types
        self.lpe = lpe
        self.chunk_size = chunk_size
        self.scale_method = scale_method
        self.is_mlp = gnn in [False, None, 'None', 'none']
        self.n_msg_layers = None if self.is_mlp else layers
        self.upd_scale = upd_scale

        if self.edge_types > 0:
            self.layer_embed = nn.Embedding(self.edge_types, hid)

        out_dim = seq_len if self.dms else 1
        mlp_kwargs = {'hid_dim': hid, 'n_layers': 2, 'act_name': act_name}
        self.edge_proj = MLP(in_dim=(1 if self.is_mlp else self.max_feat_size) * ctx,
                             **dict(mlp_kwargs, n_layers=1 if input_layer == 'linear' else 2))
        if self.is_mlp:
            self.edge_mlp = MLP(in_dim=hid,
                                out_dim=out_dim,
                                **dict(mlp_kwargs, n_layers=4))
        else:

            if self.wte_pos_enc:
                self.wte_pos_enc_layer = nn.Embedding(vocab_size + 1, hid)  # +1 for dummy token

            if self.lpe or not self.wte_pos_enc or not self.improved_graph:
                self.node_proj = MLP(in_dim=max(1, self.lpe + int(1 - self.improved_graph) * self.max_feat_size * ctx),
                                     **dict(mlp_kwargs, n_layers=1 if input_layer == 'linear' else 2))

            final_edge_update = kwargs.pop('final_edge_update', False)
            self.gnn = PNA(in_channels=hid,
                           hidden_channels=hid,
                           num_layers=self.n_msg_layers,
                           out_channels=hid,
                           act=act_name,
                           aggregators=['mean'],
                           scalers=['identity'],
                           update_edge_attr=True,
                           modulate_edges=True,
                           gating_edges=False,
                           final_edge_update=final_edge_update,
                           edge_dim=hid,
                           norm=None,
                           chunk_size=chunk_size,
                           message_passing_device=message_passing_device,
                           edge_sample_ratio=edge_sample_ratio,
                           **kwargs)

            self.edge_out = MLP(in_dim=hid,
                                out_dim=self.max_feat_size * out_dim,
                                **mlp_kwargs)

            if not self.improved_graph:
                self.node_out = MLP(in_dim=hid,
                                    out_dim=self.max_feat_size * out_dim,
                                    **mlp_kwargs)

        self.initializer_range = 0.02
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, pyg.nn.dense.linear.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                 pyg.nn.norm.layer_norm.LayerNorm, pyg.nn.norm.batch_norm.BatchNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        else:
            assert not hasattr(module, 'weight') and not hasattr(module, 'bias'), type(module)

    def fw_split(self, module, x, inplace=False):
        """
        Forward pass of the model with splitting of the input tensor for better memory efficiency.
        :param module: nn.Module
        :param x: input tensor with the first dimension to be sliced
        :param inplace: whether to perform the operation inplace
        :return: output tensor
        """
        if self.training:
            return module(x)
        if not inplace:
            x_out = []
        chunk_size = len(x) if self.chunk_size in [0, -1, None] else self.chunk_size
        for i in range(0, len(x), chunk_size):
            if inplace:
                x[i:i + chunk_size] = module(x[i:i + chunk_size])
            else:
                x_out.append(module(x[i:i + chunk_size]))
        return x if inplace else torch.cat(x_out, dim=0)

    def forward(self, graphs, k=None):
        """
        Forward pass of the model.
        :param graphs: pytorch geometric batch of graphs (can be multiple models stacked as disconnected graphs)
        :param k: number of steps to predict into the future (only used for dms during inference)
        :return: graphs with updated edge (and node) features
        """

        graphs.edge_attr = graphs.edge_attr.unflatten(1, (-1, self.ctx))
        max_feat_size = graphs.edge_attr.shape[1]
        if self.residual:
            edge_attr_res = graphs.edge_attr[:, :, self.ctx - 1]  # last parameter values

        # if the input is on cpu, move it to cuda by chunks to accommodate large models during inference
        p = next(self.edge_proj.parameters())
        device = p.device
        dtype = p.dtype
        if self.training or p.device == graphs.edge_attr.device:
            device = None

        if self.is_mlp:

            # By using chunking in th MLP, we avoid storing the full edge_attr tensor (n_params, feat_dim) in memory
            chunk_size = len(graphs.edge_attr) if self.chunk_size in [0, -1, None] or self.training else self.chunk_size

            if self.dms and not self.training:
                assert k is not None and k >= 1, k
                # avoid predicting for all the future steps during inference to save computation
                # only predict for k-th step
                fc = self.edge_mlp.fc
                w = fc[-1].weight.data.clone()
                b = fc[-1].bias.data.clone()
                fc[-1].weight.data = fc[-1].weight.data[k - 1:k]
                fc[-1].bias.data = fc[-1].bias.data[k - 1]
                fc[-1].out_features = 1
            else:
                fc = self.edge_mlp.fc

            for i in range(0, len(graphs.edge_attr), chunk_size):
                graphs.edge_attr[i:i + chunk_size, :, :1] = fc(self.edge_proj(
                    graphs.edge_attr[i:i + chunk_size].to(device).to(dtype)) + (
                    self.layer_embed(graphs.edge_type[i:i + chunk_size].to(device)).unsqueeze(1)
                    if self.edge_types else 0))
            graphs.edge_attr = graphs.edge_attr[:, :, :1]

            if self.dms and not self.training:
                fc[-1].weight.data = w
                fc[-1].bias.data = b

        else:
            edge_types = self.layer_embed(graphs.edge_type.to(device)) if self.edge_types else 0

            x_lpe = self.fw_split(self.node_proj, graphs.pos.to(device)) if self.lpe else 0
            wte_pos_emb = self.wte_pos_enc_layer(graphs.pos_w.to(device)) if self.wte_pos_enc else 0
            if self.lpe:
                assert x_lpe.dim() == 2, x_lpe.shape
            if self.wte_pos_enc:
                assert wte_pos_emb.dim() == 2, wte_pos_emb.shape
            graphs.x = wte_pos_emb + x_lpe
            graphs.edge_attr = graphs.edge_attr.flatten(1, 2)
            assert graphs.x.dim() == graphs.edge_attr.dim() == 2, (graphs.x.shape, graphs.edge_attr.shape)

            if self.training:
                graphs.edge_attr = edge_types + self.edge_proj(graphs.edge_attr.to(device).to(dtype))
            else:
                if max_feat_size < self.max_feat_size:
                    fc = self.edge_proj.fc
                    fc[0].weight.data = fc[0].weight.data[:, :max_feat_size * self.ctx]
                    fc[0].in_features = max_feat_size * self.ctx

                chunk_size = len(graphs.edge_attr) if self.chunk_size in [0, -1, None] else self.chunk_size
                if self.edge_types:
                    for i in range(0, len(graphs.edge_attr), chunk_size):
                        edge_types[i:i + chunk_size] = self.edge_proj(
                            graphs.edge_attr[i:i + chunk_size].to(device).to(dtype)) + edge_types[i:i + chunk_size]
                else:
                    edge_types = self.edge_proj(graphs.edge_attr)

                graphs.edge_attr = edge_types
                del edge_types

            graphs.x, graphs.edge_attr = self.gnn(
                x=graphs.x, edge_index=graphs.edge_index.to(device), edge_attr=graphs.edge_attr)

            if self.dms and not self.training:
                assert k is not None and k >= 1, k

                fc = self.edge_out.fc
                _, in_dim = fc[-1].weight.shape
                n_out = max_feat_size
                w = fc[-1].weight.data.clone()
                b = fc[-1].bias.data.clone()

                fc[-1].weight.data = fc[-1].weight.data.reshape(self.max_feat_size, -1, in_dim)[:n_out, k - 1]
                fc[-1].bias.data = fc[-1].bias.data.reshape(self.max_feat_size, -1)[:n_out, k - 1]
                fc[-1].out_features = n_out

                graphs.edge_attr = self.fw_split(fc, graphs.edge_attr).unsqueeze(2)
                fc[-1].weight.data = w
                fc[-1].bias.data = b
            else:
                graphs.edge_attr = self.fw_split(self.edge_out, graphs.edge_attr).unflatten(
                    1, (self.max_feat_size, -1))

        if self.residual:
            graphs.edge_attr = edge_attr_res.unsqueeze(-1) + graphs.edge_attr.to(edge_attr_res) * self.upd_scale

        if self.training:
            graphs.edge_attr = graphs.edge_attr.flatten(1, 2)

        return graphs


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, n_layers=1, act_name='silu'):
        super().__init__()
        hid_dim = hid_dim or in_dim
        out_dim = out_dim or hid_dim
        layers = []
        for layer in range(n_layers):
            in_dim_ = in_dim if layer == 0 else hid_dim
            out_dim_ = out_dim if layer == n_layers - 1 else hid_dim
            layers.append(nn.Linear(in_dim_, out_dim_))
            if layer < n_layers - 1:
                layers.append(activation_resolver(act_name))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
