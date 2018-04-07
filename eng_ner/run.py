import nltk
import sys

sys.path.append('./eng_ner_tagger')

from configs import *
import json
from torch.autograd import Variable
from model.Bilstm import BiLSTM
import torch

class NE_Tagger():

    def __init__(self):
        self.vocab_size = None
        self.pos_vocab_size = None
        self.output_dim = None
        self._set_data()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layer
        self.embedding_dim = embedding_dim

        self.model = BiLSTM(self.vocab_size,
                            self.pos_vocab_size,
                            self.hidden_dim,
                            self.num_layers,
                            self.embedding_dim,
                            self.output_dim)
        self.model.load_state_dict(torch.load(model_path,map_location=lambda storage,loc:storage))


    def _set_data(self):

        json_data = open(vocab_file_path).read()
        self.vocab_set = json.loads(json_data)
        self.vocab_size = len(self.vocab_set['word2idx'])
        self.pos_vocab_size = len(self.vocab_set['pos2idx'])
        self.output_dim = len(self.vocab_set['tag2idx'])
        self.idx_tag = self.vocab_set['idx2tag']
        self.idx_tag.pop('0')

    def tokenize(self,input_text):

        text = input_text.split(' ')
        tmp = nltk.pos_tag(text)

        text_batch = []
        pos_batch = []

        for item,pos in tmp:
            text_batch.append(item)
            pos_batch.append(pos)


        return text_batch,pos_batch

    def prepare_sequence(self,seq,pos_seq,word2idx,pos2idx):

        word_idxs = []
        pos_idxs = []
        for word in seq:
            if word not in word2idx:
                word_idxs.append(word2idx['<unk>'])
            else:
                word_idxs.append(word2idx[word])

        for pos in pos_seq:
            if pos not in pos2idx:
                pos_idxs.append(pos2idx['<unk>'])
            else:
                pos_idxs.append(pos2idx[pos])

        return word_idxs,pos_idxs



    def tag(self,input_str):

        self.model.eval()
        text_batch,pos_batch = self.tokenize(input_str)

        input,pos_input = self.prepare_sequence(text_batch,pos_batch,self.vocab_set['word2idx'],self.vocab_set['pos2idx'])

        #input = Variable(torch.from_numpy(np.array([input])),volatile=False)
        #pos_input = Variable(torch.from_numpy(np.array([pos_input])), volatile=False)

        input = Variable(torch.LongTensor([input]))
        pos_input = Variable(torch.LongTensor([pos_input]))

        hidden_state = self.model.init_hidden(1)

        output,logits = self.model(input,pos_input,hidden_state)

        max_predictions, argmax_predictions = output.max(2)

        tagged_list = argmax_predictions.cpu().data.numpy()[0]

        output_list = []
        for i in range(len(tagged_list)):
            tag = str(tagged_list[i])
            if tag=='0':
                output_list.append('O')
            else:
                output_list.append(self.idx_tag[tag])

        return text_batch,output_list





