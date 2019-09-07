import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
import torch.nn.functional as F
import networkx as nx

from dgl.utils import Index
from dgl.batched_graph import batch
from dgl.graph import  DGLGraph
# from dgl.
from gat import GAT







def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size)).cuda()


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        # the adj mat
        self.W = nn.Parameter(torch.empty(input_size, input_size).uniform_(0, 1), requires_grad=False)
        # self

        self.lstm_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        # self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)
        #
        self.final_linear = nn.Linear(in_features=input_size, out_features=1)
        #
        # self.v_e = nn.Parameter(torch.randn(1, (self.T-1)), requires_grad=True)
        self.encode = nn.Linear(1, input_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = ([2] * 1) + [1]

        self.gat = GAT(num_layers=1,
                       in_dim=(T-1),

                       num_hidden=hidden_size,
                       num_classes=1,
                       heads=heads,
                       activation=F.elu,
                       feat_drop=0.6,
                       attn_drop=0.6,
                       negative_slope=0.2,
                       residual=False
                       )
        # del nx_graph
        self.gat.cuda()

    def forward(self, input_data):

        # input_data: (batch_size, T - 1, input_size)
        # LSTM

        # _, new_data = self.lstm_layer(torch.matmul(self.W, input_data.transpose(2, 1)).transpose(1, 2).transpose(1, 0))
        # input_weighted  = new_data.squeeze(0)
        nx_graph = nx.barabasi_albert_graph(self.input_size, 3)
        # nx_graph = nx.from_numpy_matrix(self.W.detach().cpu().numpy())


        gs = DGLGraph(nx_graph)
        gs.to(self.device)  # for _ in range(input_data.shape[0])]
        #
        input_weighted = torch.zeros(input_data.size(0), self.input_size).cuda()


        # # g_batch = batch(gs)
        # # g_batch.to(self.device)
        for i in range(input_data.shape[0]):
            input_weighted[i] = self.gat(gs, torch.tanh(torch.matmul(self.W , input_data[i].transpose(1, 0))))[:, 0]
            # gs.clear()
            # gs = DGLGraph(nx_graph.copy())
            # gs.to(self.device)  # for _ in range(input_data.shape[0])]
        #

        #     gs
        return self.final_linear(input_weighted)
        # return input_weighted, input_encoded
