"""
数据预处理部分
"""
from torchtext import data
from torchtext.data import Field, Example, TabularDataset, Dataset
from string import punctuation


class InputData(object):
    def __init__(self, data_file, max_length=300):
        self.stopword = [f for f in punctuation]
        self.file = [line.strip() for line in open(data_file, 'r', encoding='utf8')]
        self.TEXT = data.Field(sequential=True, use_vocab=True, batch_first=True, stop_words=self.stopword,
                               fix_length=max_length)
        self.LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True, unk_token=None)

    def create_iter(self, split_ratio, batch_size=100):
        fields = [("text", self.TEXT), ("label", self.LABEL)]
        examples = []
        for index, context in enumerate(self.file):
            d = context.split('\t')
            # item = [text, label]
            item = [d[1], d[0].strip()]
            examples.append(data.Example().fromlist(item, fields))
        train, valid, test = Dataset(examples=examples, fields=fields).split(split_ratio=split_ratio)
        self.TEXT.build_vocab(train)
        self.LABEL.build_vocab(train)
        voca_size = len(self.TEXT.vocab)
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (train, valid, test), sort_key=lambda x: len(x.text),
            batch_sizes=(batch_size, len(valid), len(test)))
        return train_iter, val_iter, test_iter, voca_size


def main():
    data = InputData('../data/corpus.txt', max_length=500)
    train_iter, val_iter, test_iter, voca = data.create_iter(split_ratio=[0.8, 0.1, 0.1], batch_size=100)
    print(len(train_iter))
    print(len(val_iter))
    print(len(test_iter))
    print(voca)
    print(data.LABEL.vocab.stoi)


if __name__ == '__main__':
    main()
