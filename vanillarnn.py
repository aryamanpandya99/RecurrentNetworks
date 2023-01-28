'''
Aryaman Pandya 
Sequential Machine Learning 
Building a Vanilla RNN 
Model and trainer implementation 
'''
import torch 
import plotly
from torch import nn

class VanillaRNN(nn.module): 
    
    def __init__(self, len_in, len_h, len_out) -> None:
        super(VanillaRNN, self).__init__()
        self.len_h = len_h #size of hidden state
        self.in_h = nn.Linear(len_in + len_h, len_h) #graph module to compute next hidden state 
        self.in_out = nn.Linear(len_in + len_h, len_out) #computes output 

    def forward(self, x, h):
        combined = torch.cat((x, h), 1)
        h_out = nn.ReLU(self.in_h(combined)) 
        y = self.in_out(combined)
        return y, h_out 

    def init_h(self):
        return torch.zeros(1, self.len_h)


def train(model, dataset, loss_function, optim, epochs, device):
    losses = []
    
    for epoch in range(epochs):
        print("Epoch %d / %d" % (epoch+1, epochs))
        print("-"*10)
    
        for i, (x, y) in enumerate(dataset):
            h_s = model.init_h()
            for dp in dataset:
                y_pred, h_out = model('''Enter inputs here''')
            loss = loss_function(y_pred, y)
            losses.append(loss)
        
            optim.zero_grad()
            loss.backward()
            optim.step()

            if(i % 10):
              print("Step: {}/{}, current Epoch loss: {:.4f}".format(i, len(dataset), loss))  




