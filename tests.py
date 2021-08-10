# As suggested by [cite slideshow] we decided to follow the common prototyping paradigm of test-driven development [cite]
# Thereby, every functionality of the project is supposed to be required first by a dedicated test fixture.
# Tests are first implemented in simple fashion in tests.py. Once such a failing test fixture is conceived 
# we set about implementing code to satisfy its conditions. The intent is to keep iterating over test-code
# and related production-code until the desired functionality is achieved.

import unittest
import math
import data, model, run
import spacy
import tensorflow_datasets as tfds

TEST_CORPUS_EN = """The final project should implement a system 
        related to deep learning for NLP using the Py- Torch library 
        and test it. The project is documented in an ACL-style paper 
        that adheres to the standards of practice in computational 
        linguistics."""

TEST_CORPUS_DA = """Det endelige projekt skal implementere et system
        relateret til dyb læring for NLP ved hjælp af Py-Torch-biblioteket
        og test det. Projektet er dokumenteret i et papir i ACL-stil
        der overholder standarderne for praksis inden for beregning
        lingvistik."""

class DataFixture(unittest.TestCase):

    def test_dataset_builder(self):
        builder = data.get_para_crawl_builder('da')
        M = "Make sure to have tensorflow_datasets installed."
        self.assertIsInstance(builder, tfds.core.DatasetBuilder, msg=M)
        builder.download_and_prepare()
        train, test, val = data.get_data_splits(builder)
        TOTAL = len(builder.as_dataset(split='train'))
        M2 = "Amount of examples should correspond roughly to the desired splits."
        self.assertAlmostEqual(len(train), int(0.7 * TOTAL), delta=1, msg=M2)
        self.assertAlmostEqual(len(test), int(0.2 * TOTAL), delta=1, msg=M2)
        self.assertAlmostEqual(len(val), int(0.1 * TOTAL), delta=1, msg=M2)

    def test_vocabulary(self):
        test_vocab = data.Vocab(TEST_CORPUS_EN)
        # run some test on our implementation of vocabulary
        self.assertEqual(test_vocab.get_word(0), '<S>')
        self.assertEqual(test_vocab.get_word(3), 'The')
        self.assertEqual(test_vocab.get_word(5), 'project')
        self.assertEqual(test_vocab.get_id('The'), 3)
        self.assertEqual(test_vocab.get_id('project'), 5)
        self.assertEqual(test_vocab.get_count('the'), 2)
        self.assertEqual(test_vocab.get_count('project'), 2)

    def test_spacy_option(self):
        english_nlp = spacy.load('en_core_web_sm')
        test_vocab = data.Vocab(TEST_CORPUS_EN, spacy_nlp=english_nlp)
        M = "Make sure to have required SpaCy packages installed."
        self.assertEqual(test_vocab.get_word(0), '<S>')
        self.assertEqual(test_vocab.get_word(3), 'The')
        self.assertEqual(test_vocab.get_word(5), 'project')
        self.assertEqual(test_vocab.get_id('The'), 3)
        self.assertEqual(test_vocab.get_id('project'), 5)
        self.assertEqual(test_vocab.get_count('the'), 2)
        self.assertEqual(test_vocab.get_count('project'), 2)

    def test_vocab_filter(self):
        test_vocab = data.Vocab(TEST_CORPUS_EN)
        top_30 = test_vocab.slice(n_samples=30)
        self.assertEqual(len(top_30), 30)

class RunFixture(unittest.TestCase):

    def test_another(self):
        self.assertAlmostEqual(1,1)

    def test_more(self):
        self.assertAlmostEqual(1,1)

if __name__ == '__main__':
    unittest.main(verbosity=2)