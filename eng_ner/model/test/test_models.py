import sys

sys.path.append('../')
from Embedding import Embedding
from Bilstm import BiLSTM
import torch
from torch.autograd import Variable

import numpy as np

'''
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
'''

def test_embedding():
    a = Embedding(10,3)
    input = Variable(torch.LongTensor(2, 4))
    input1 = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
    output = a(input)
    assert output.size() == (2, 4, 3)


def test_answer():
    batch_size, seq_size, hidden_dim = 10, 20, 50
    vocab_size, emb_dim = 100, 30
    num_layers = 2

    embedding = Embedding(vocab_size, emb_dim)
    model = BiLSTM(emb_dim, hidden_dim, num_layers, embedding, output_dim=vocab_size)

    np_inputs = np.random.randint(0, vocab_size-1, (batch_size, seq_size))
    inputs = Variable(torch.LongTensor(np_inputs))

    hidden_states = model.init_hidden(batch_size)
    outputs, logits = model(inputs, hidden_states)
    print(outputs.size())
