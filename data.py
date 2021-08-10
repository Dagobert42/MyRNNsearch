import re
from collections import OrderedDict
import tensorflow_datasets as tfds

# the chosen dataset holds sentence pairs in the following form: show example
# As the current ParaCrawl configuration contains only a train split we subdivide
# this training set further to obtain test and validation splits.

# one possible reason for the performance differences between .. and oru network 
# is the size of the datasets that were used for training
# [cite Bahdanau] train expansively on [training numbers]

######################_HELPERS_######################

def get_para_crawl_builder(target_language):
    builder = tfds.builder("para_crawl", config=f'en{target_language}')
    return builder

def get_data_splits(builder, train=70, test=20):
    d = str(train + test)
    val = 100 - train - test
    train_split = builder.as_dataset(split=f'train[:{train}%]')
    test_split = builder.as_dataset(split=f'train[{train}%:{d}%]')
    val_split = builder.as_dataset(split=f'train[-{val}%:]')
    return train_split, test_split, val_split

######################_DATA_CLASSES_######################

class Vocab:
    """
    A vocabulary in the form of an ordered dictionary.
    """
    class Entry:
        def __init__(self, id):
            """ An entry to the vocabulary. With index and count. """
            self.id = id
            self.count = 1

        def __repr__(self):
            """ String prepresentation for printing. """
            return str((self.id, self.count))

    def __init__(self, text=None, spacy_nlp=None):
        """
        Creates a vocabulary over an input text in the form of
        an ordered dictionary containing indeces and word counts.
        Set use_spacy to True to make use of a SpaCy for tokenization.
        """
        self.SPECIALS = '<S> <E> <U>'
        self.words = OrderedDict()
        self.spacy_nlp = spacy_nlp

        # add special tokens to vocabulary
        for word in self.SPECIALS.split():
            id = len(self.words.keys())
            self.words[word] = self.Entry(id)

        if text:
            self.append(text)

    def append(self, txt):
        """ Adds a string token by token to the vocabulary. """
        # use SpaCy for tokenization if requested
        if self.spacy_nlp:
            for tok in self.spacy_nlp.tokenizer(txt):
                word = tok.text
                if tok.text not in self.words.keys():
                    next_id = len(self.words.keys())
                    self.words[word] = self.Entry(next_id)
                else:
                    self.words[word].count += 1
        else:
            for word in txt.split():
                if word not in self.words.keys():
                    next_id = len(self.words.keys())
                    self.words[word] = self.Entry(next_id)
                else:
                    self.words[word].count += 1

    def get_word(self, id):
        """ Access utility. """
        # this might be somewhat slow but will only
        # be called when translating from ids to words
        return list(self.words.items())[id][0]

    def get_id(self, word):
        """ Access utility. """
        return self.words[word].id

    def get_count(self, word):
        """ Access utility. """
        return self.words[word].count

    def slice(self, n_samples, descending=True):
        """ Returns a slice of this vocabs dictionary sorted by word count. """
        sorted_list = list(sorted(
                self.words.items(),
                key=lambda item: item[1].count,
                reverse=descending))
        return OrderedDict({k: v for k, v in sorted_list[:n_samples]})
