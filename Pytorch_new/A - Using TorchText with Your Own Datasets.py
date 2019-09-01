# @TIME : 2019/7/31 上午00:35
# @File : A - Using TorchText with Your Own Datasets.py

"""
 define the Fields
 loaded the dataset
 created the splits


 TEXT = data.Field()
 LABEL = data.LabelField()

 train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
 train_data, valid_data = train_data.split()


 three type Torchtext can read: json(best), tsv, csv

"""

# Reading json
import torch
from torchtext import data
from torchtext import datasets

NAME = data.Field()
SAYING = data.Field()
PLACE = data.Field()

fields = {'name':('n', NAME), 'location':('p', PLACE), 'quote':('s', SAYING)}


train_data, test_data = data.TabularDataset.splits(
    path='data',
    train = 'train.json',
    test = 'test.json',
    format = 'json',
    fields = fields
)
a= 1

print(vars(train_data[0]))
NAME.build_vocab(train_data)
SAYING.build_vocab(train_data)
PLACE.build_vocab(train_data)
a = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, test_itertor = data.BucketIterator.splits(
    (train_data, test_data),
    sort = False,
    batch_size=1,
    device = device
)

print('train')
for batch in train_iterator:
    print(batch)

print()
print('test')
for batch in test_itertor:
    print(batch)


"""
结果

{'n': ['John'], 'p': ['United', 'Kingdom'], 's': ['i', 'love', 'the', 'united kingdom']}
train

[torchtext.data.batch.Batch of size 1]
	[.n]:[torch.LongTensor of size 1x1]
	[.p]:[torch.LongTensor of size 2x1]
	[.s]:[torch.LongTensor of size 4x1]
32

test

[torchtext.data.batch.Batch of size 1]
	[.n]:[torch.LongTensor of size 1x1]
	[.p]:[torch.LongTensor of size 2x1]
	[.s]:[torch.LongTensor of size 4x1]

"""