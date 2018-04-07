import numpy as np
import torch
from tqdm import tqdm
import json
from model.Bilstm import BiLSTM
from torch import nn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from preprocess import get_iterator,get_dataset
from configs import *

train,val = get_dataset()


num_epochs = 50
batch_size= 32

vocab_size = len(train.fields['src'].vocab.stoi)
pos_vocab_size = len(train.fields['pos'].vocab.stoi)
output_dim = len(train.fields['tgt'].vocab.stoi)
'''
vocab_set = {
    'word2idx' : {},
    'pos2idx' : {},
    'tag2idx' : {},
    'idx2tag' : {}
    }

vocab_set['word2idx'] = dict(train.fields['src'].vocab.stoi)
vocab_set['pos2idx'] = dict(train.fields['pos'].vocab.stoi)
vocab_set['tag2idx'] = dict(train.fields['tgt'].vocab.stoi)


idx2tag = {}
for idx,item in enumerate(train.fields['tgt'].vocab.itos):
    idx2tag[idx] = item

vocab_set['idx2tag'] = idx2tag

with open('vocab_set.json','w') as file:
    json.dump(vocab_set,file)
    print('file saved in directory')

'''

model = BiLSTM(vocab_size,pos_vocab_size,hidden_dim,num_layer,embedding_dim,output_dim=output_dim).cuda()

criterion = nn.NLLLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train(True)

max_f1_score = 0
# Training
for epoch in range(num_epochs):

    train_generator,val_generator = get_iterator(train,val,32)

    for step,item in tqdm(enumerate(train_generator)):

        inputs = item.src
        pos_input = item.pos
        targets = item.tgt


        input_batch_size = inputs.size(0)

        hidden_state = model.init_hidden(input_batch_size)

        model.zero_grad()

        output,logits = model(inputs,pos_input,hidden_state)
        seq_length = output.size(1)

        max_predictions,argmax_predictions = output.max(2)

        loss = 0

        for i in range(seq_length):
            loss += criterion(logits[:, i, :], targets[:, i])

        loss.backward()

        #torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        if (step + 1) % 100 == 0:
            acc = []
            f1_measure = []
            for i in range(input_batch_size):
                acc.append(accuracy_score(targets.cpu().data.numpy()[i],argmax_predictions.cpu().data.numpy()[i]))
                f1_measure.append(f1_score(targets.cpu().data.numpy()[i],argmax_predictions.cpu().data.numpy()[i],average='micro'))
            print ('Epoch [%d/%d], Step: %d, Loss: %.3f, Acc: %.3f, f1 score: %.3f, Perplexity: %5.2f' %
                   (epoch+1, num_epochs, step, loss.data[0] / batch_size,np.mean(acc),np.mean(f1_measure), np.exp(loss.data[0]/ batch_size)))


    if (epoch + 1) % 1 == 0:
        model.eval()
        for step, item in tqdm(enumerate(val_generator)):
            input = item.src
            pos_input = item.pos
            target = item.tgt

            input_batch_size = input.size(0)

            hidden_state = model.init_hidden(input_batch_size)

            output, logits = model(input, pos_input, hidden_state)
            seq_length = output.size(1)

            max_predictions, argmax_predictions = output.max(2)

            val_loss = 0

            for i in range(seq_length):
                val_loss += criterion(logits[:, i, :], target[:, i])

            if (step + 1) % 20 == 0:
                val_acc = []
                val_f1_measure = []
                for i in range(input_batch_size):
                    val_acc.append(
                        accuracy_score(target.cpu().data.numpy()[i], argmax_predictions.cpu().data.numpy()[i]))
                    val_f1_measure.append(
                        f1_score(target.cpu().data.numpy()[i], argmax_predictions.cpu().data.numpy()[i],
                                 average='micro'))

                print(
                        'Epoch [%d/%d], Step: %d, val Loss: %.3f, val Acc: %.3f, val f1 score: %.3f, val_Perplexity: %5.2f' %
                        (epoch + 1, num_epochs, step, val_loss.data[0] / batch_size, np.mean(val_acc),
                         np.mean(val_f1_measure),
                         np.exp(val_loss.data[0] / batch_size)))

                if max_f1_score < np.mean(val_acc):
                    if step>30 :
                        max_f1_score = np.mean(val_acc)
                        print('the max f1 score is : ',np.mean(val_acc))
                        torch.save(model.state_dict(),'ner_model_f1_max.pkl')

        model.train()