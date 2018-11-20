import torch
import torch.optim as optim
import torch.nn as nn

class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        node = 128
        self.relu = torch.nn.ReLU()
        self.batch = torch.nn.BatchNorm1d(node, momentum=0.01)
        self.dropout = torch.nn.Dropout(p=1 - 0.9)

        self.input_layer = torch.nn.Linear(44, node, bias=True)
        self.linear = torch.nn.Linear(node, node, bias=True)

        self.hidden_layer = torch.nn.Sequential(self.linear, self.batch, self.dropout, self.relu,
                                                self.linear, self.batch, self.relu,
                                                self.linear, self.batch, self.dropout, self.relu,
                                                self.linear, self.batch, self.dropout, self.relu,
                                                self.linear, self.batch, self.dropout, self.relu,
                                                self.linear, self.batch, self.relu,
                                                self.linear, self.batch, self.relu,
                                                self.linear, self.batch, self.dropout, self.relu,
                                                self.linear, self.batch, self.dropout, self.relu,
                                                self.linear, self.batch, self.relu,
                                                self.linear, self.batch, self.dropout, self.relu,
                                                self.linear, self.batch, self.dropout, self.relu,
                                                self.linear, self.batch, self.relu
                                                )

        self.output_layer = torch.nn.Linear(node, 3, bias=True)

        torch.nn.init.kaiming_uniform_(self.input_layer.weight)
        torch.nn.init.kaiming_uniform_(self.linear.weight)

        torch.nn.init.kaiming_uniform_(self.output_layer.weight)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)

        return out