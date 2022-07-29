import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True, dropout=0.2):
        super(LSTM, self).__init__()

        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first



        self.rnn = nn.LSTM(self.input_size,
                           self.hidden_size,
                           self.num_layers,
                           self.bidirectional,
                           self.batch_first)
        if self.bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1



       # self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, args, x):
        h_0 = Variable(torch.zeros(self.num_layers * self.num_direction, int(args.batch), args.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.num_layers * self.num_direction, int(args.batch), args.hidden_size).cuda())
       
       
        #x = x.view(int(args.batch),x.size(1),args.hidden_size * 2)
        x_packed, (h, c) = self.rnn(x)

        return x_packed


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
          
        x = self.linear(x)
        #print('x2',x)
        return x

