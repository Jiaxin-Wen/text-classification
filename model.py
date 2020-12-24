import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, vocab, input_size, hidden_size, fc_size, num_layers, cell,dropout):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_size)
        self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        
        if cell == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                #dropout=dropout,
                bias=True,
                #batch_first = True,
                bidirectional=True,
            )
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(
                input_size =input_size,
                hidden_size = hidden_size,
                num_layers = num_layers,
                #dropout = dropout,
                bias=True,
                bidirectional =True
            )
        
        self.fc1 = nn.Linear(
            in_features=hidden_size * 2, 
            out_features=8,
            bias=True
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        output,h = self.rnn(x)
        x = output[-1]
        x = self.fc1(x)
        return x


class AttentionRNN(nn.Module):
    def __init__(self, vocab, input_size, hidden_size, fc_size, num_layers,cell, dropout):
        super(AttentionRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_size)
        self.embedding.weight.data.copy_(vocab.vectors)
        self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.w = nn.Parameter(torch.randn((input_size*2,1)))
        
        if cell =='GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                #dropout=dropout,
                bias=True,
                batch_first = True,
                bidirectional=True,
            )
        
        elif cell =='LSTM':
            self.rnn = nn.LSTM(
                input_size =input_size,
                hidden_size = hidden_size,
                num_layers = num_layers,
                #dropout = dropout,
                bias=True,
                bidirectional =True
            )
    
        self.fc = nn.Linear(
            in_features=hidden_size*2,
            out_features=8,  #8种情感
            bias=True
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        output,h = self.rnn(x)
        output = torch.transpose(output, 1,2)

        m = torch.tanh(output)
        a = F.softmax(torch.matmul(self.w.t(),m), dim=2)   #w: []
        r = torch.matmul(output, torch.transpose(a,1,2))
        r = r.squeeze(2)
        r = torch.tanh(r)
        x = self.fc(r)
        return x


class CNN(nn.Module):
    def __init__(self, vocab, input_size,fix_length, dropout):
            super(CNN, self).__init__()
            self.fix_length = fix_length
            self.embedding = nn.Embedding(len(vocab), input_size)
            self.embedding.weight.data.copy_(vocab.vectors)
            self.embedding.weight.requires_grad = True  #默认是false　不参与训练
            
            self.filter_sizes=[3,4,5]
            self.features = 100#300
            self.convs = nn.ModuleList([nn.Conv2d(1,self.features,(k, input_size) )for k in self.filter_sizes])

            self.dropout = nn.Dropout(dropout)
            
            self.fc = nn.Linear(self.features * len(self.filter_sizes), 8)

    def conv_and_pool(self,x,conv):
        x = conv(x)
        x = F.relu(x).squeeze(3)
        x = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self,x):
        
            x = self.embedding(x)
            x = x.unsqueeze(1)
            x = torch.cat([self.conv_and_pool(x,conv) for conv in self.convs],1)
            x = self.dropout(x)
            x = self.fc(x)
            return x


class MLP(nn.Module):
    def __init__(self, vocab, input_size,fix_length, dropout):
            super(MLP, self).__init__()
            self.fix_length = fix_length
            self.input_size = input_size
            self.embedding = nn.Embedding(len(vocab), input_size)
            self.embedding.weight.data.copy_(vocab.vectors)
            self.embedding.weight.requires_grad = False  #默认是false　不参与训练
            self.fc1 = nn.Linear(fix_length*input_size,512)
            self.fc2 = nn.Linear(512,32)
            self.dropout = nn.Dropout(dropout)
            self.fc3 = nn .Linear(32,8)
    
    def forward(self,x):
            x = self.embedding(x)
            x = x.view(-1,self.fix_length*self.input_size)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x 
