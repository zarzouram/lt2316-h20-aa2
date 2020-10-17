import re
from itertools import zip_longest

from torch.utils.data import Dataset

class Vocabulary:
    # map word to ints
    def __init__(self, start=0):
        self.vocab = {}
        self.reverse_vocab = {}
        self.__next = start

    def add(self, word):
        # Add word to the vocabulary dict
        if word not in self.vocab:
            self.vocab[word] = self.__next
            self.reverse_vocab[self.__next] = word
            self.__next += 1
        return self.vocab.get(word)

    def __getitem__(self, item):
        # Get the word id of given word or vise versa
        if isinstance(item, int):
            return self.reverse_vocab.get(item, None)
        else:
            return self.vocab.get(item, None)

    def __len__(self):
        # Get vocabulary size
        return len(self.vocab)


class DatasetObject(Dataset):
    # construct pytorch dataset
    def __init__(self, df):
        self._df = df
        self.construct_ds(df)

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        labels = self.labels[idx]
        length = self.lengthes[idx]
        loc = self.loc[idx]
        return list(zip_longest(tokens, labels, loc, [length]))
        # self.sent_id[idx,:]

    def construct_ds(self, df):
        # sort df by seq lengthes, to reduce number of pad tokens
        df.sort_values(by=["length"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # construct dataset fields
        self.labels = DatasetField("labels", df)
        self.tokens = DatasetField("tokens", df)
        self.lengthes = DatasetField("length", df)
        self.loc = DatasetField("loc", df)
        # self.sent_id = DatasetField(df"sent_id"].to_list())


class DatasetField(DatasetObject):
    def __init__(self, field, df=None):
        self._data = None
        self.__initiate_field(field, df)

    def __initiate_field(self, field, df):
        self._data = df[field]

    def __getitem__(self, idx=None):
        return self._data[idx]