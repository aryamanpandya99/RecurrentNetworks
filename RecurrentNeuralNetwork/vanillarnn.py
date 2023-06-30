'''
Aryaman Pandya 
Sequential Machine Learning 
Building a Vanilla RNN 
Model and trainer implementation 
'''
import torch 
import pandas as pd
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import plotly
from torchtext.datasets import AG_NEWS
from torch import nn
from torch.utils.data import Dataset, DataLoader

#Class definition of Vanilla RNN 
class VanillaRNN(nn.Module): 
    
    def __init__(self, vocab_size, embed_size, input_len, hidden_size, output_len, num_layers) -> None:
        super(VanillaRNN, self).__init__()
        
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.hidden_size = hidden_size 
        self.input_len = input_len 
        self.output_len = output_len 
        self.rnn = nn.RNN(nput_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=True) #graph module to compute next hidden state 
        
        self.hidden2label = nn.Linear(2*hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropoutLayer = nn.Dropout()

    def forward(self, x, text_len):
        embedded = self.encoder(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len)
        output, hidden = self.rnn(packed_embedded)  # Pass the initial hidden state 'h' to the RNN
        
        
        # Flatten the output tensor to match the linear layer input size
        output = output.contiguous().view(-1, 2 * self.hidden_size)
        
        # Apply dropout to the flattened output
        output = self.dropoutLayer(output)
        
        # Pass the output through the linear layer
        output = self.hidden2label(output)
        
        # Apply softmax activation to get probabilities
        output = self.softmax(output)
        
        return output, hidden

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

def collate_batch(batch):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list, text_list, offsets = [], [], [0]
    
    vocab = build_vocab_from_iterator(yield_tokens(batch), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    tokenizer = get_tokenizer("basic_english")
    label_pipeline = lambda x: int(x) - 1
    text_pipeline = lambda x: vocab(tokenizer(x))
    
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    
    return label_list.to(device), text_list.to(device), offsets.to(device)

def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

#execution 
def main():
    
    train_iter = (AG_NEWS(split="train"))
    train, test = AG_NEWS()
    print(train)
    '''print(next(train_iter))
    
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"]) # specials allows us to handle out of distribution tokens 
    vocab.set_default_index(vocab["<unk>"]) # set out of distrubution tokens by default to specials 
    
    print(vocab(['testing', 'our', 'tokenization', 'thishastobeoutofdistribution'])) #last two terms are out of distro and get assigned the same index '''
    
    train_loader = DataLoader(train_iter, batch_size = 8, shuffle = True, collate_fn = collate_batch)
    print(train_loader.dataset.data)
    
    RANDOM_SEED = 123
    torch.manual_seed(RANDOM_SEED)

    VOCABULARY_SIZE = 5000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    DROPOUT = 0.5
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    EMBEDDING_DIM = 128
    BIDIRECTIONAL = True
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    OUTPUT_DIM = 4
    
    model = VanillaRNN()
    
    



if __name__=="__main__":
    main()



