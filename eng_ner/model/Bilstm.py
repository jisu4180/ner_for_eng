import sys
sys.path.append('./eng_ner_tagger')

import torch
from torch.nn import functional as F
from torch import nn
from model.Embedding import Embedding
from torch.autograd import Variable

class BiLSTM(nn.Module):

    def __init__(self,
                vocab_size,
                pos_vocab_size,
                hidden_dim,
                num_layers,
                embedding_dim,
                output_dim=None,
                dropout=0.2):

        super(BiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.pos_vocab_size = pos_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size,embedding_dim)
        self.pos_embedding = Embedding(pos_vocab_size,10)


        self.rnn = nn.LSTM(embedding_dim + 10,
                            self.hidden_dim,
                            num_layers,
                            dropout,
                            batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        if output_dim is not None :
            self.output_dim = output_dim
            self.linear = nn.Linear(2*hidden_dim,output_dim)


    def init_hidden(self,batch_size):
        h_n = Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_dim))
        c_n = Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_dim))
        return(h_n,c_n)


    def forward(self,vocab,pos,hidden_state):

        embed_vector = self.embedding(vocab)
        pos_embed_vector = self.pos_embedding(pos)
        enhanced_embedding = torch.cat((embed_vector,pos_embed_vector),-1)

        hidden_vector,_ = self.rnn(enhanced_embedding, hidden_state)

        if self.output_dim is not None:
            output_vector = self.linear(self.dropout(hidden_vector))


        logit = F.log_softmax(output_vector, dim=-1)

        return output_vector, logit
