from run import NE_Tagger

from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report
from itertools import chain

validation_set = pd.read_csv('./data/dev/dataset_val_renew.csv')

sentence = list(validation_set.iloc[:, 0])
answer = list(validation_set.iloc[:, 2])

predictions = []
tars = []


tagger = NE_Tagger()

for i,line in tqdm(enumerate(sentence)):
    if line[0] == ' ':
        pass
    elif line[len(line)-1]==' ':
        pass
    else:
        sen,pred = tagger.tag(line)
        tar = answer[i].split(' ')
        predictions.append(pred)
        tars.append(tar)

print('summary report')

print("==============================================================================")

print(classification_report(list(chain(*tars)), list(chain(*predictions))))

print('==============================================================================')
