"""clssifier based on RNN hiddens"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class sntClassifier_2tails(nn.Module):
    '''sentence classification'''
        
    def __init__(self, hidden_dim, class_size, bidirectional=True, num_layers=1, dropout=0., device=None, multi_class=False):
        """Initialize model."""
        super(sntClassifier_2tails, self).__init__()
        self.hidden_dim = hidden_dim
        self.class_size = class_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.multi_class = multi_class
        self.hidden_layers = 1

        self.num_directions = 2 if self.bidirectional else 1
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # The linear layer that maps from hidden state space to tag space
        if self.hidden_layers == 1:
            self.hidden2class = nn.Linear(self.num_directions * self.hidden_dim, self.class_size)
        else:
            self.hidden2class = nn.Sequential(
                    nn.Linear(self.num_directions * self.hidden_dim, self.hidden_dim),
                    nn.Sigmoid(),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(self.hidden_dim, self.class_size)
                    )
            #self.hidden2class = [nn.Linear(self.num_directions * self.hidden_dim, self.hidden_dim)]
            #for i in range(1, self.hidden_layers-1):
            #    self.hidden2class.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            #self.hidden2class.append(nn.Linear(self.hidden_dim, self.class_size))

        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        for weight in self.hidden2class.parameters():
            weight.data.uniform_(-initrange, initrange)
        
    def forward(self, packed_h_t_c_t, masked_output=None):
        index_slices = [self.num_layers*2-2, self.num_layers*2-1] if self.bidirectional else [self.num_layers*1-1]
        index_slices = torch.tensor(index_slices, dtype=torch.long, device=self.device)
        h_t = torch.index_select(packed_h_t_c_t[0], 0, index_slices)
        h_t = h_t.transpose(0,1)
        h_t = h_t.contiguous().view(h_t.size(0), self.num_directions*h_t.size(2))
        class_space = self.hidden2class(self.dropout_layer(h_t))
        if self.multi_class:
            class_scores = torch.sigmoid(class_space)
            if type(masked_output) != type(None):
                class_scores.index_fill_(1, masked_output, 0)
        else:
            class_scores = F.log_softmax(class_space, dim=1)
        
        return class_scores
    
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

class sntClassifier_hiddenPooling(nn.Module):
    '''sentence classification'''
        
    def __init__(self, hidden_dim, class_size, bidirectional=True, num_layers=1, dropout=0., device=None, multi_class=False, pooling='mean'):
        """Initialize model."""
        super(sntClassifier_hiddenPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.class_size = class_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.pooling = pooling
        self.multi_class = multi_class

        assert self.pooling in ('max', 'mean')

        self.num_directions = 2 if self.bidirectional else 1
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden_layers = 2 #1
        if self.hidden_layers == 1:
            self.hidden2class = nn.Linear(self.num_directions * self.hidden_dim, self.class_size)
        else:
            activation = nn.Sigmoid()
            h2c = [nn.Linear(self.num_directions * self.hidden_dim, self.hidden_dim), activation, nn.Dropout(p=self.dropout)]
            for i in range(self.hidden_layers - 2):
                h2c += [nn.Linear(self.hidden_dim, self.hidden_dim), activation, nn.Dropout(p=self.dropout)]
            h2c.append(nn.Linear(self.hidden_dim, self.class_size))
            self.hidden2class = nn.Sequential(*h2c)
            '''self.hidden2class = nn.Sequential(
                    nn.Linear(self.num_directions * self.hidden_dim, self.hidden_dim),
                    nn.Sigmoid(),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(self.hidden_dim, self.class_size)
                    )
            '''

        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        for weight in self.hidden2class.parameters():
            weight.data.uniform_(-initrange, initrange)
        
    def forward(self, inputs, masked_output=None):
        '''
        lstm_out : bsize x seqlen x hsize
        '''
        lstm_out, lens = inputs
        if self.pooling == 'mean':
            len_sum = torch.tensor(lens, dtype=torch.float, device=self.device).unsqueeze(1)
            lstm_out_pool = lstm_out.sum(1).squeeze(1)
            lens_sum = len_sum.repeat(1, lstm_out.size(2))
            lstm_out_pool = lstm_out_pool / lens_sum
        elif self.pooling == 'max':
            max_len = max(lens)
            mask = [
                ([0] * l) + ([1] * (max_len - l))
                for l in lens
                ]
            mask = torch.tensor(mask, dtype=torch.uint8, device=self.device)
            mask = mask.unsqueeze(2).repeat(1, 1, lstm_out.size(2))
            #lstm_out.data.masked_fill_(mask, -float('inf'))
            lstm_out_pool = lstm_out.max(1)[0].squeeze(1)
        class_space = self.hidden2class(self.dropout_layer(lstm_out_pool))
        if self.multi_class:
            class_scores = torch.sigmoid(class_space)
            if type(masked_output) != type(None):
                class_scores.index_fill_(1, masked_output, 0)
        else:
            class_scores = F.log_softmax(class_space, dim=1)
        
        return class_scores
    
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

class sntClassifier_hiddenCNN(nn.Module):
    '''sentence classification'''
        
    def __init__(self, hidden_dim, class_size, bidirectional=True, num_layers=1, dropout=0., device=None, multi_class=False, kernel_size=3):
        """Initialize model."""
        super(sntClassifier_hiddenCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.class_size = class_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.multi_class = multi_class
        self.kernel_size = kernel_size

        self.num_directions = 2 if self.bidirectional else 1
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.batchnorm = nn.BatchNorm1d(self.num_directions * self.hidden_dim)

        self.conv2d = False
        if not self.conv2d:
            self.kernel_size = 3 #7
            self.cnn = nn.Sequential(
                    nn.Conv1d(self.num_directions * self.hidden_dim, self.num_directions * self.hidden_dim, self.kernel_size, padding=1), #, padding=self.kernel_size//2),
                    nn.ReLU(),
                    #nn.Dropout(p=self.dropout),
                    #nn.Conv1d(self.num_directions * self.hidden_dim, self.num_directions * self.hidden_dim, 3, padding=self.kernel_size//2),
                    #nn.ReLU(),
                    )
            '''self.cnn2 = nn.Sequential(
                    nn.Conv1d(self.num_directions * self.hidden_dim, self.num_directions * self.hidden_dim, 3, self.kernel_size, padding=1), #padding=self.kernel_size//2),
                    nn.ReLU(),
                    )
            self.cnn3 = nn.Sequential(
                    nn.Conv1d(self.num_directions * self.hidden_dim, self.num_directions * self.hidden_dim, 5, self.kernel_size, padding=1), #padding=self.kernel_size//2),
                    nn.ReLU(),
                    )
            '''
        else:
            self.cnn = nn.Sequential(
                    nn.Conv2d(1, 1, (7, 7), padding=(3, 3)),
                    nn.ReLU(),
                    )
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden_layers = 1 #1
        if self.hidden_layers == 1:
            self.hidden2class = nn.Linear(self.num_directions * self.hidden_dim, self.class_size)
        else:
            activation = nn.Sigmoid()
            h2c = [nn.Linear(self.num_directions * self.hidden_dim, self.hidden_dim), activation, nn.Dropout(p=self.dropout)]
            for i in range(self.hidden_layers - 2):
                h2c += [nn.Linear(self.hidden_dim, self.hidden_dim), activation, nn.Dropout(p=self.dropout)]
            h2c.append(nn.Linear(self.hidden_dim, self.class_size))
            self.hidden2class = nn.Sequential(*h2c)
            '''self.hidden2class = nn.Sequential(
                    nn.Linear(self.num_directions * self.hidden_dim, self.hidden_dim),
                    nn.Sigmoid(),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(self.hidden_dim, self.class_size)
                    )
            '''

        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        for weight in self.cnn.parameters():
            weight.data.uniform_(-initrange, initrange)
        '''for weight in self.cnn2.parameters():
            weight.data.uniform_(-initrange, initrange)
        for weight in self.cnn3.parameters():
            weight.data.uniform_(-initrange, initrange)
        '''
        for weight in self.hidden2class.parameters():
            weight.data.uniform_(-initrange, initrange)
        
    def forward(self, inputs, masked_output=None):
        '''
        lstm_out : bsize x seqlen x hsize
        '''
        lstm_out, lens = inputs
        hiddens = lstm_out.transpose(1, 2)
        hiddens = self.dropout_layer(hiddens)
        if not self.conv2d:
            conv_hiddens_1 = self.cnn(hiddens)
            #conv_hiddens_2 = self.cnn3(hiddens)
            #conv_hiddens_3 = self.cnn2(hiddens)
            conv_hiddens_pool_1 = conv_hiddens_1.max(2)[0]
            #conv_hiddens_pool_2 = conv_hiddens_2.max(2)[0]
            #conv_hiddens_pool_3 = conv_hiddens_3.max(2)[0]
            ##conv_hiddens_pool = torch.cat((conv_hiddens_pool_1, conv_hiddens_pool_2, conv_hiddens_pool_3), 1)
            conv_hiddens_pool = conv_hiddens_pool_1 #+ conv_hiddens_pool_2 + conv_hiddens_pool_3
        else:
            hiddens = hiddens.unsqueeze(1)
            conv_hiddens_1 = self.cnn(hiddens)
            conv_hiddens_1 = conv_hiddens_1.squeeze(1)
            conv_hiddens_pool_1 = conv_hiddens_1.max(2)[0]
            conv_hiddens_pool = conv_hiddens_pool_1
        conv_hiddens_pool = self.batchnorm(conv_hiddens_pool)
        class_space = self.hidden2class(self.dropout_layer(conv_hiddens_pool))
        if self.multi_class:
            class_scores = torch.sigmoid(class_space)
            if type(masked_output) != type(None):
                class_scores.index_fill_(1, masked_output, 0)
        else:
            class_scores = F.log_softmax(class_space, dim=1)
        
        return class_scores
    
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

class sntClassifier_hiddenAttention(nn.Module):
    '''sentence classification'''
        
    def __init__(self, hidden_dim, class_size, bidirectional=True, num_layers=1, dropout=0., device=None, multi_class=False):
        """Initialize model."""
        super(sntClassifier_hiddenAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.class_size = class_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.multi_class = multi_class

        self.num_directions = 2 if self.bidirectional else 1
        
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.Wa = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        #self.Wa2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.Ua = nn.Conv1d(self.num_directions * self.hidden_dim, self.hidden_dim, 1, bias=False)
        self.Va = nn.Conv1d(self.hidden_dim, 1, 1, bias=False)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2class = nn.Linear(self.num_directions * self.hidden_dim, self.class_size)

        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        for m in (self.Wa, self.Ua, self.Va):#, self.Wa2):
            m.weight.data.uniform_(-initrange, initrange)
        self.hidden2class.weight.data.uniform_(-initrange, initrange)
        self.hidden2class.bias.data.uniform_(-initrange, initrange)
        
    def forward(self, inputs, masked_output=None):
        '''
        lstm_out : bsize x seqlen x hsize
        '''
        packed_h_t_c_t, lstm_out, lens = inputs
        enc_h_t, enc_c_t = packed_h_t_c_t
        if self.bidirectional:
            index_slices = [2 * self.num_layers - 1]  # generated from the reversed path
            index_slices = torch.tensor(index_slices, dtype=torch.long, device=self.device)
            h_t = torch.index_select(enc_h_t, 0, index_slices)
            c_t = torch.index_select(enc_c_t, 0, index_slices)
        else:
            h_t = enc_h_t
            c_t = enc_c_t

        hiddens = lstm_out.transpose(1, 2)
        h_t = h_t.squeeze(0)
        c1 = self.Wa(self.dropout_layer(h_t))
        #c1 = self.Wa2(self.dropout_layer(c1))
        c2 = self.Ua(self.dropout_layer(hiddens))

        c3 = c1.unsqueeze(2).repeat(1, 1, lstm_out.size(1))
        c4 = torch.tanh(c3 + c2)

        e = self.Va(c4).squeeze(1)
        max_len = max(lens)
        mask = [
            ([0] * l) + ([1] * (max_len - l))
            for l in lens
            ]
        mask = torch.tensor(mask, dtype=torch.uint8, device=self.device)
        e.masked_fill_(mask, -float('inf'))
        a = F.softmax(e, dim=1)

        context_hidden = torch.bmm(hiddens, a.unsqueeze(2)).squeeze(2)

        class_space = self.hidden2class(self.dropout_layer(context_hidden))
        if self.multi_class:
            class_scores = torch.sigmoid(class_space)
            if type(masked_output) != type(None):
                class_scores.index_fill_(1, masked_output, 0)
        else:
            class_scores = F.log_softmax(class_space, dim=1)
        
        return class_scores
    
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

