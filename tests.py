import unittest
import math
import data, model, run
import torch
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
        self.assertEqual(test_vocab.ids[0], '<S>')
        self.assertEqual(test_vocab.ids[3], 'The')
        self.assertEqual(test_vocab.ids[5], 'project')
        self.assertEqual(test_vocab.words['The'].id, 3)
        self.assertEqual(test_vocab.words['project'].id, 5)
        self.assertEqual(test_vocab.words['the'].count, 2)
        self.assertEqual(test_vocab.words['project'].count, 2)

    def test_spacy_option(self):
        english_nlp = spacy.load('en_core_web_sm')
        test_vocab = data.Vocab(TEST_CORPUS_EN, spacy_nlp=english_nlp)
        M = "Make sure to have required SpaCy packages installed."
        self.assertEqual(test_vocab.ids[0], '<S>')
        self.assertEqual(test_vocab.ids[3], 'The')
        self.assertEqual(test_vocab.ids[5], 'project')
        self.assertEqual(test_vocab.words['The'].id, 3)
        self.assertEqual(test_vocab.words['project'].id, 5)
        self.assertEqual(test_vocab.words['the'].count, 2)
        self.assertEqual(test_vocab.words['project'].count, 2)

    def test_vocab_filter(self):
        test_vocab = data.Vocab(TEST_CORPUS_EN)
        top_30 = test_vocab.slice(n_samples=30)
        self.assertEqual(len(test_vocab.words), 30)
        self.assertEqual(len(top_30.words), 30)
        self.assertEqual(top_30, test_vocab)

    def test_sentence_to_indeces(self):
        test_vocab = data.Vocab(TEST_CORPUS_EN)
        vec = test_vocab.get_indeces("This is a test")
        self.assertListEqual(vec, [2, 24, 8, 22, 1])
        sent = test_vocab.get_sentence(vec)
        self.assertEqual(sent, '<U> is a test <E>')

class ModelFixture(unittest.TestCase):

    def test_encoder(self):
        dummy = model.Encoder(10, 10, 10)

    def test_guess_anything(self):
        test_tensor = torch.zeros(30, dtype=torch.long).view(-1, 1)
        dummy = model.RNNsearch(10, 10, 10, 10, 10)
        out = dummy(test_tensor, test_tensor, 0)
        self.assertIsInstance(out, torch.Tensor)

    def test_init_weights(self):
        pass

    def test_get_single_translation(self):
        test_tensor = torch.zeros(30, dtype=torch.long).view(-1, 1)
        dummy = model.RNNsearch(10, 10, 10, 10, 10)
        out = dummy(test_tensor, test_tensor, 0)
        self.assertIsInstance(out, torch.Tensor)

class RunFixture(unittest.TestCase):

    def test_batching(self):
        pass

    def test_evaluate_model(self):
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)