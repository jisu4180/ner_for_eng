import torch
from torch.nn import functional as F
from torch import nn



class Embedding(nn.Module):

    def __init__(self,vocab_size,embed_size):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.vocab_size,self.embed_size)

    def get_vocab_size(self):
        return self.vocab_size

    def forward(self,input):

        output = self.embedding(input)
        return output
