# NER for Engilsh Model

## Entity Tagger (BiLSTM)

### env
    * Python 3.6.1
    * Pytorch 0.6.0
    * nltk

## 1. Input ,Output

### Input

    for training

    data with source,pos,target split by token

    example

    src) Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country

    pos) NNS IN NNS VBP VBN IN NNP TO VB DT NN IN NNP CC VB DT NN IN JJ NNS IN DT NN

    tgt) O O O O O O B-geo O O O O O B-geo O O O O O B-gpe O O O O


    for test

    just run code with row data

    example

    Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country


### Output

  NE token (list)

  ex)

  input> Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country

  output> [O,O,O,O,O,O,B-geo,O,O,O,O,O,B-geo,O,O,O,O,O,B-gpe,O,O,O,O]


## 2. project path

### data source

    https://www.kaggle.com/nltkdata/conll-corpora

### path

    * model path : ./data/model/ner_model_f1_max.pkl
    * data path : ./data/dev/dataset_train.csv


## 3. How to Train

    * python train.py
______________________________
### Params

    model = BiLSTM(vocab_size = data vocabulary size,
                        pos_vocab_size = data pos vocabulary size,
                        hidden_dim = dimension of hidden,
                        num_layers = dimension of layers,
                        embedding_dim = embedding dimension,
                        output_dim = label_size)

______________________________
### 4. How to Predict

    from run import NE_Tagger

    tagger = NE_Tagger

    tagger.tag('your_input_str')

______________________________
### load pretrained model

    self.model.load_state_dict(torch.load(model_path,map_location=lambda storage,loc:storage))


### Author

    Jisoo Kim

    last commit : 18/04/05
