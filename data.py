import re
import tensorflow_datasets as tfds
import torch

######################_HELPERS_######################

def get_para_crawl_builder(target_language):
    return tfds.builder("para_crawl", config=f'en{target_language}_plain_text')

def get_data_splits(builder, train=70, test=20):
    d = str(train + test)
    val = 100 - train - test
    train_split = builder.as_dataset(split=f'train[:{train}%]')
    test_split = builder.as_dataset(split=f'train[{train}%:{d}%]')
    val_split = builder.as_dataset(split=f'train[-{val}%:]')
    return train_split, test_split, val_split

######################_DATA_CLASS_######################

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
        self.spacy_nlp = spacy_nlp
        self.words = {}
        self.ids = {}

        # add special tokens to vocabulary
        for word in self.SPECIALS.split():
            id = len(self.words.keys())
            self.words[word] = self.Entry(id)
            self.ids[id] = word

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
                    self.ids[next_id] = word
                else:
                    self.words[word].count += 1
        else:
            for word in txt.split():
                if word not in self.words.keys():
                    next_id = len(self.words.keys())
                    self.words[word] = self.Entry(next_id)
                    self.ids[next_id] = word
                else:
                    self.words[word].count += 1

    def filter(self, n_samples, descending=True):
        """ Reduces this vocabs dictionary to n_samples after sorting by word count. """
        sorted_list = list(sorted(
                self.words.items(),
                key=lambda item: item[1].count,
                reverse=descending))
        self.words = {k: v for k, v in sorted_list[:n_samples]}
        return self

    def get_indeces(self, sentence):
        """ Produces a representation from the indeces in this vocabulary. """
        STA = 0
        END = 1
        UNK = 2
        if self.spacy_nlp:
            seq = [self.words[word].id if word in self.words.keys() else UNK
                for word in self.spacy_nlp.tokenizer(sentence)]
        else:
            seq = [self.words[word].id if word in self.words.keys() else UNK
                for word in sentence.split()]
        seq.append(END)
        return seq
    
    def get_sentence(self, indeces):
        """
        Converts a list of indeces into a readable sentence
        using words from this vocabulary.
        """
        return ' '.join([self.ids[id] for id in indeces])

