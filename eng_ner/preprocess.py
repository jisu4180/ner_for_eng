from torchtext import data


'''
dataset = torchtext.data.TabularDataset(
            path = './data/dataset.csv',
            format = 'csv',
            fields = [('src',data.Field()),('pos',data.Field()),('tgt',data.Field)]
)


def prepare_csv(seed=9999,ratio=0.2):

    df_train = pd.read_csv("dataset.csv")

    idx = np.arange(df_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)
    val_size = int( len(idx) * ratio)

    df_train.iloc[idx[val_size:],:].to_csv(
        "./data/dev/dataset_train.csv",index=False
    )
    df_train.iloc[idx[:val_size],:].to_csv(
        "./data/dev/dataset_val.csv",index=False
    )

def tokenizer(text):
    return text.split(' ')

prepare_csv()
'''

def get_dataset():

    src = data.Field(batch_first=True)
    pos = data.Field(batch_first=True)
    tgt = data.Field(batch_first=True)

    train,val = data.TabularDataset.splits(
        path = './data/dev/',format='csv',skip_header=True,
        train ='dataset_train_renew.csv',validation = 'dataset_val_renew.csv',
        fields = [
            ('src',src),
            ('pos',pos),
            ('tgt',tgt)])


    src.build_vocab(train.src,min_freq =3)
    pos.build_vocab(train.pos)
    tgt.build_vocab(train.tgt)

    return train,val

def get_iterator(train,val,batch_size):

    '''
    train_iter,val_iter = data.BucketIterator(
        (train,val),batch_size=batch_size,sort_within_batch=True,shuffle=True,
        sort_key = lambda x: data.interleave_keys(len(x.src),len(x.pos),len(x.tgt)),device=-1
    )
    '''
    train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_within_batch=True, repeat=False, shuffle=True,
                    sort_key= lambda x: data.interleave_keys(len(x.src),len(x.tgt)))

    val_iter = data.BucketIterator(dataset=val, batch_size=64, sort_within_batch=True, repeat=False,
                                     shuffle=True,sort_key=lambda x: data.interleave_keys(len(x.src), len(x.tgt)))

    train_generator = train_iter.__iter__()
    val_generator = val_iter.__iter__()

    return train_generator,val_generator
