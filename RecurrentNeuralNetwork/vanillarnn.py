'''
Aryaman Pandya 
Sequential Machine Learning 
Building a Vanilla RNN 
Model and trainer implementation 
'''
import torch 
import pandas as pd
import numpy as np
import plotly
from torch import nn

#Class definition of Vanilla RNN 
class VanillaRNN(nn.Module): 
    
    def __init__(self, len_in, len_h, len_out) -> None:
        super(VanillaRNN, self).__init__()
        self.len_h = len_h #size of hidden state
        self.in_h = nn.Linear(len_in + len_h, len_h) #graph module to compute next hidden state 
        self.in_out = nn.Linear(len_in + len_h, len_out) #computes output 

    def forward(self, x, h):
        combined = torch.cat((x, h), 1)
        h_out = nn.Tanh(self.in_h(combined)) 
        y = self.in_out(combined)
        return y, h_out 

    def init_h(self):
        return torch.zeros(1, self.len_h)

#trainer function for our RNN 
def train(model, dataset, loss_function, optim, epochs, device):
    losses = [] #group losses for loss visualization 
    
    for epoch in range(epochs):
        print("Epoch %d / %d" % (epoch+1, epochs))
        print("-"*10)
    
        for i, (x, y) in enumerate(dataset):
            h_s = model.init_h() #initialize hidden state 
            for dp in dataset:
                y_pred, h_out = model('''Enter inputs here''')
            loss = loss_function(y_pred, y) 
            losses.append(loss)
        
            optim.zero_grad()
            loss.backward() #backprop 
            optim.step() #update weights

            if(i % 10):
              print("Step: {}/{}, current Epoch loss: {:.4f}".format(i, len(dataset), loss))  

#splits dataset into training and testing sets 
def test_train_split(df):
    
    mask = np.random.rand(len(df)) < 0.8
    train = df[mask]
    test = df[~mask]

    return train[['Date', 'Close']], test[['Date', 'Close']] 

def batched_data(df, n):
    x, y = [], []
    for i in range(len(df)-n):
        tmp = df[i:(i+n), :]
        x.append(tmp)
        y.append(df[i+n,:])
    return np.array(x), np.array(y)


#execution 
def main():
    #load IBM stonk data 
    ibm_df = pd.read_csv('IBM.csv')
    train_set, test_set = test_train_split(ibm_df)

    x, y = batched_data(train_set, 10)

    print(train_set)
    print(x)




if __name__=="__main__":
    main()



