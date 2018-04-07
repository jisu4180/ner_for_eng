from run import NE_Tagger
import pandas as pd
from sklearn.metrics import classification_report
tagger = NE_Tagger()
from itertools import chain


#print(tagger.tag('Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country'))


validation_set = pd.read_csv('./data/dev/dataset_val_renew.csv')

sentence = list(validation_set.iloc[:,0])
answer = list(validation_set.iloc[:,2])

i = 0

predictions = []
tars = []

for i,line in enumerate(sentence):
    if line[0] == ' ':
        pass
    else:
        sen,pred = tagger.tag(line)
        tar = answer[i].split(' ')
        predictions.append(pred)
        tars.append(tar)

print('summary report')
print(classification_report(list(chain(*tars)),list(chain(*predictions))))