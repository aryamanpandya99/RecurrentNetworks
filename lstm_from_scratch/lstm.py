"""
LSTM class implementation
"""
import torch 
from torch import nn

class lstm(nn.Module): 
    def __init__(self, input, hidden_dim) -> None:
        super().__init__()
        self.forget_gate = nn.Sequential(
            nn.Linear(input+hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.input_gate = nn.Sequential(
            nn.Linear(input+hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.input_node = nn.Sequential(
            nn.Linear(input+hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Linear(input+hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, h_in, c_in):
        x_h = torch.cat((x, h_in), 2)
        i_gate_output = self.input_gate(x_h)
        i_node_output = self.input_node(x_h)
        o_gate_output = self.output_gate(x_h)
        f_gate_output = self.forget_gate(x_h)

        c_out = (f_gate_output * c_in) + (i_node_output * i_gate_output)

        h_out = nn.Tanh(c_out) * o_gate_output

        return h_out, c_out
    
    def init_h(self):
        return torch.zeros(1, self.hidden_size)
