from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from tell import LogicalLayer, Phi

import torch

from torch import Tensor


def gumbel_sigmoid(logits, tau = 1, hard = False, threshold = 0.5, deterministic=False):
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    if deterministic: gumbels=0
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class GumbelSigmoidLayer(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False, threshold=0.5, deterministic=False):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.threshold = threshold
        self.deterministic = deterministic

    def forward(self, logits):
        return gumbel_sigmoid(
            logits,
            tau=self.tau,
            hard=self.hard,
            threshold=self.threshold,
            deterministic=self.deterministic
        )
    
    def set(self, tau, hard, deterministic):
        """Permette di aggiornare la temperatura a runtime"""
        self.tau = tau
        self.hard = hard
        self.deterministic = deterministic

from torch_geometric.nn import MessagePassing

#First Version
# class CustomGraphConv(MessagePassing):
#     def __init__(self, nn, eps=0., train_eps=False, **kwargs):
#         super().__init__(aggr='add', **kwargs)
#         self.nn = nn
#         self.initial_eps = eps
#         if train_eps:
#             self.eps = torch.nn.Parameter(torch.Tensor([eps]))
#         else:
#             self.register_buffer('eps', torch.Tensor([eps]))
        
#     def forward(self, x, edge_index, edge_attr):
        
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
#         node_feat_dim = x.size(1)
#         edge_feat_dim = edge_attr.size(1)
#         node_part = out[:, :node_feat_dim]  
#         edge_part = out[:, node_feat_dim:node_feat_dim + edge_feat_dim]  
#         node_part_with_self = (1 + self.eps) * x + node_part
#         out = torch.cat([node_part_with_self, edge_part], dim=1)
#         out = self.nn(out)

#         return out
    
#     def message(self, x_j, edge_attr):
#         return torch.cat([x_j, edge_attr], dim=1)
    
#     def update(self, aggr_out):
#         return aggr_out

#Second Version
class CustomGraphConv(MessagePassing):
    def __init__(self, nn_0, nn_1, negative_concatenate=False, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.nn_0 = nn_0
        self.nn_1 = nn_1
        self.messages = None
        self.negative_concatenate = negative_concatenate

    def forward(self, x, edge_index, edge_attr):
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = self.nn_1(out)
        #print("out shape after nn_1:", out.shape)
        return out, self.messages
    
    def message(self, x_i, x_j, edge_attr):
        #print("x_i shape:", x_i.shape, "x_j shape:", x_j.shape, "edge_attr shape:", edge_attr.shape if edge_attr is not None else None)
        if edge_attr is None:
            out = torch.cat([x_i, x_j], dim=1)
        else:
            out = torch.cat([x_i, x_j, edge_attr], dim=1)
        #print("out shape before nn_0:", out.shape)
        out = self.nn_0(out)
        #print("out shape after nn_0:", out.shape)
        self.messages = out
        if self.nn_0.__class__.__name__ == "LogicalLayer" and self.negative_concatenate == 2:
            out = torch.cat([out, 1-out], dim=1)
        return out

    def update(self, aggr_out):
        return aggr_out

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_features_edge, num_classes, num_layers=3, hidden_dim=64, dropout=0.15, nogumbel=False, edge_once=False, layer_double = False):
        super(GIN, self).__init__()
        self.num_features, self.num_features_edge, self.num_classes = num_features, num_features_edge, num_classes
        self.convs = torch.nn.ModuleList()
        self.edge_once = edge_once
        self.layer_double = layer_double
        for i in range(num_layers):

            in_dim = None
            if i == 0:
                in_dim = 2 * num_features + num_features_edge
            elif edge_once:
                in_dim = 2 * hidden_dim
            else:
                in_dim = 2 * hidden_dim + num_features_edge
            
            if layer_double == False:
                conv = CustomGraphConv(
                    nn_0 = torch.nn.Sequential(
                        torch.nn.Linear(in_dim, hidden_dim),
                        GumbelSigmoidLayer() if not nogumbel else torch.nn.ReLU()
                    ),
                    nn_1 = torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        GumbelSigmoidLayer() if not nogumbel else torch.nn.ReLU()
                    )
                )
            else:
                conv = CustomGraphConv(
                    nn_0 = torch.nn.Sequential(
                        torch.nn.Linear(in_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        GumbelSigmoidLayer() if not nogumbel else torch.nn.ReLU()
                    ),
                    nn_1 = torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        GumbelSigmoidLayer() if not nogumbel else torch.nn.ReLU()
                    )
                )
            self.convs.append(conv)
        self.fc1 = torch.nn.Linear(num_layers*3*hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.nogumbel=nogumbel

    def forward(self, x, edge_index, edge_attr, batch=None, tau=1, deterministic=False, *args, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0]).long().to(x.device)
        xs = []
        for i, conv in enumerate(self.convs):

            if self.layer_double == False and not self.nogumbel:
                conv.nn_0[1].set(tau, hard=True, deterministic=deterministic)
                conv.nn_1[1].set(tau, hard=True, deterministic=deterministic)
            elif self.layer_double == True and not self.nogumbel:
                conv.nn_0[3].set(tau, hard=True, deterministic=deterministic)
                conv.nn_1[3].set(tau, hard=True, deterministic=deterministic)

            if i == 0 and edge_attr is not None:
                x, _ = conv(x, edge_index, edge_attr)
            elif i!=0 and edge_attr is not None and self.edge_once == False:
                x, _ = conv(x, edge_index, edge_attr)
            else:
                x, _ = conv(x, edge_index, None)

            xs.append(x)
            x = self.dropout(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def forward_e(self, x, edge_index, edge_attr, batch=None, tau=1, deterministic=False, *args, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0]).long().to(x.device)
        ret_x = []
        ret_y_interm = []
        ret_y = []
        xs = []
        for i, conv in enumerate(self.convs):

            if self.layer_double == False and not self.nogumbel:
                conv.nn_0[1].set(tau, hard=True, deterministic=deterministic)
                conv.nn_1[1].set(tau, hard=True, deterministic=deterministic)
            elif self.layer_double == True and not self.nogumbel:
                conv.nn_0[3].set(tau, hard=True, deterministic=deterministic)
                conv.nn_1[3].set(tau, hard=True, deterministic=deterministic)

            ret_x.append(x)

            if i == 0 and edge_attr is not None:
                x, x_interm = conv(x, edge_index, edge_attr)
            elif i!=0 and edge_attr is not None and self.edge_once == False:
                x, x_interm = conv(x, edge_index, edge_attr)
            else:
                x, x_interm = conv(x, edge_index, None)

            xs.append(x)
            ret_y.append(x)
            ret_y_interm.append(x_interm)
            x = self.dropout(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        ret_x.append(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.sigmoid(self.fc2(x))        
        ret_y.append(x)
        return ret_x, ret_y, ret_y_interm



class GINTELL(torch.nn.Module):
    def __init__(self, num_features, num_features_edge, num_classes, num_layers=3, hidden_dim=64, dropout=0.1, edge_again=False, negative_concatenate=False, input_binary=True):
        super(GINTELL, self).__init__()
        self.num_features, self.num_features_edge, self.num_classes = num_features, num_features_edge, num_classes
        self.convs = torch.nn.ModuleList()
        self.edge_again = edge_again
        self.negative_concatenate = negative_concatenate
        print("edge_again", edge_again, "negative_concatenate", negative_concatenate)
        if not edge_again and negative_concatenate==2:
            for i in range(num_layers):
                conv = CustomGraphConv(
                    nn_0 = LogicalLayer(2*2*num_features + 2*num_features_edge if i==0 else 2*2*hidden_dim, hidden_dim, use_phi=False),
                    nn_1 = LogicalLayer(2*hidden_dim, hidden_dim, use_phi=True),
                    negative_concatenate=negative_concatenate
                )   
                self.convs.append(conv)
            self.fc = LogicalLayer(2*num_layers*3*hidden_dim, num_classes, use_phi=True)
        elif edge_again and negative_concatenate==0:
            for i in range(num_layers):
                conv = CustomGraphConv(
                    nn_0 = LogicalLayer(2*num_features + num_features_edge if i==0 else 2*hidden_dim + num_features_edge, hidden_dim, use_phi=False),
                    nn_1 = LogicalLayer(hidden_dim, hidden_dim, use_phi=True),
                    negative_concatenate=negative_concatenate
                )   
                self.convs.append(conv)
            self.fc = LogicalLayer(num_layers*3*hidden_dim, num_classes, use_phi=True)
        elif edge_again and negative_concatenate==2:
            for i in range(num_layers):
                conv = CustomGraphConv(
                    nn_0 = LogicalLayer(2*2*num_features + 2*num_features_edge if i==0 else 2*2*hidden_dim + 2*num_features_edge, hidden_dim, use_phi=False),
                    nn_1 = LogicalLayer(2*hidden_dim, hidden_dim, use_phi=True),
                    negative_concatenate=negative_concatenate
                )   
                self.convs.append(conv)
            self.fc = LogicalLayer(2*num_layers*3*hidden_dim, num_classes, use_phi=True)
        elif not edge_again and negative_concatenate==0:
            for i in range(num_layers):
                conv = CustomGraphConv(
                    nn_0 = LogicalLayer(2*num_features + num_features_edge if i==0 else 2*hidden_dim, hidden_dim, use_phi=False),
                    nn_1 = LogicalLayer(hidden_dim, hidden_dim, use_phi=True),
                    negative_concatenate=negative_concatenate
                )   
                self.convs.append(conv)
            self.fc = LogicalLayer(num_layers*3*hidden_dim, num_classes, use_phi=True)
        elif edge_again and negative_concatenate==1:
            for i in range(num_layers):
                conv = CustomGraphConv(
                    nn_0 = LogicalLayer(2*2*num_features + 2*num_features_edge if i==0 else 2*hidden_dim + 2*num_features_edge, hidden_dim, use_phi=False),
                    nn_1 = LogicalLayer(hidden_dim, hidden_dim, use_phi=True),
                    negative_concatenate=negative_concatenate
                )   
                self.convs.append(conv)
            self.fc = LogicalLayer(num_layers*3*hidden_dim, num_classes, use_phi=True)
        else:
            for i in range(num_layers):
                conv = CustomGraphConv(
                    nn_0 = LogicalLayer(2*2*num_features + 2*num_features_edge if i==0 else 2*hidden_dim, hidden_dim, use_phi=False),
                    nn_1 = LogicalLayer(hidden_dim, hidden_dim, use_phi=True),
                    negative_concatenate=negative_concatenate
                )   
                self.convs.append(conv)
            self.fc = LogicalLayer(num_layers*3*hidden_dim, num_classes, use_phi=True)

    
        #self.dropout = torch.nn.Dropout(dropout)
        self.input_binary = input_binary
        print("input_binary", input_binary)
        self.phi = Phi(num_features) if not input_binary else None


    def forward(self, x, edge_index, edge_attr, batch, discrete=False, *args, **kwargs):
        #x = self.input_bnorm(x)
        xs = []
        if edge_attr is not None and self.negative_concatenate!=0:
            edge_attr = torch.hstack([edge_attr, 1-edge_attr])
        for i, conv in enumerate(self.convs):
            #if i!=0:self.dropout(x)
            if i == 0 and not self.input_binary:
                x = self.phi(x)
            if self.negative_concatenate == 2 or (self.negative_concatenate==1 and i==0):
                x = torch.hstack([x, 1-x])

            if i == 0 and edge_attr is not None:
                x, _ = conv(x, edge_index, edge_attr)
            elif i!=0 and edge_attr is not None and self.edge_again:
                x, _ = conv(x, edge_index, edge_attr)
            else:
                x, _ = conv(x, edge_index, None)

            if discrete:
                indices = (x > 0.5).nonzero(as_tuple=True)
                x_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format)
                x_hard[indices[0], indices[1]] = 1.0
                x = x_hard - x.detach() + x
            xs.append(x)
            #x = self.dropout(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        if self.negative_concatenate==2:
            x = torch.hstack([x, 1-x])
        #x = self.output_bnorm(x)
        x = self.fc(x)
        # x = self.fc(x)
        return x
    

    def forward_e(self, x, edge_index, batch, discrete=False, *args, **kwargs):
        #x = self.input_bnorm(x)
        ret_x = []
        ret_y = []
        xs = []
        if edge_attr is not None and self.negative_concatenate!=0:
            edge_attr = torch.hstack([edge_attr, 1-edge_attr])
        for i, conv in enumerate(self.convs):
            ret_x.append(x)
            #if i!=0:self.dropout(x)

            if i == 0 and not self.input_binary:
                x = self.phi(x)
            if self.negative_concatenate == 2 or (self.negative_concatenate==1 and i==0):
                x = torch.hstack([x, 1-x])

            if i == 0 and edge_attr is not None:
                x, _ = conv(x, edge_index, edge_attr)
            elif i!=0 and edge_attr is not None and self.edge_again:
                x, _ = conv(x, edge_index, edge_attr)
            else:
                x, _ = conv(x, edge_index, None)

            if discrete:
                indices = (x > 0.5).nonzero(as_tuple=True)
                x_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format)
                x_hard[indices[0], indices[1]] = 1.0
                x = x_hard - x.detach() + x
            xs.append(x)
            ret_y.append(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        if self.negative_concatenate==2:
            x = torch.hstack([x, 1-x])
        #x = self.output_bnorm(x)
        ret_x.append(x)
        x = self.fc(x, discrete_output=False)
        # x = self.fc(x)
        ret_y.append(x)
        return ret_x, ret_y
