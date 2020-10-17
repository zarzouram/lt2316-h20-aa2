import re
import random
from itertools import chain
import pandas as pd
import numpy as np

from .data_loading import DataLoader, pad_post_sequence
from .custom_classes import Vocabulary, DatasetObject
from .ppmi_embedding import ppmi_embedding

import torch
from torchnlp.word_to_vector import GloVe
# Feel free to add any new code to this script


def extract_features(data: pd.DataFrame, max_sample_length: int,
                     dataset: DataLoader):
    # this function should extract features for all samples and
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb

    
    # Vocabularies
    vocab = dataset.word_vocab.vocab
    max_len = dataset.max_sample_length
    pad_token = vocab["padpad"]

    device = dataset.device

    # get sequences
    train_seqs = dataset.train
    val_seqs = dataset.val
    test_seqs = dataset.test

    train_X = pad_post_sequence(
            train_seqs["tokens"], max_len, pad_token)
    val_X = pad_post_sequence(
            val_seqs["tokens"], max_len, pad_token)
    test_X = pad_post_sequence(
            test_seqs["tokens"], max_len, pad_token)

    train_X = torch.tensor(
            train_X, dtype=torch.long, device=device)
    
    val_X = torch.tensor(
            val_X, dtype=torch.long, device=device)
    
    test_X = torch.tensor(
            test_X, dtype=torch.long, device=device)

    return train_X, val_X, test_X


def get_input_embeddings(tokens_seq, tokens_vocab):
    
    vocab_size = tokens_vocab.__len__()

    print("Get pretrained embeddings")
    embeddings_vectors = {}
    for name_, size_ in [("6B", 100), ("840B", 300)]:
        glove_vectors = GloVe(  name=name_, 
                                dim=size_, 
                                cache="/home/guszarzmo@GU.GU.SE/resources/embeddings/glove"
                            )

        vocab_vectors = np.zeros((vocab_size, size_))

        # if token is OOV (not in embeddings = words_vector) get ppmi
        for token, id_ in tokens_vocab.vocab.items():
            if not (vocab_vectors[id_, :]==0).all():
                continue
            else:
                vector = glove_vectors[token]
                if not (vector==0).all():
                    vocab_vectors[id_, :] = vector
                else:
                    vocab_vectors[id_, :] = [
                        random.uniform(-1.5, 1.7) for _ in range(size_)]
        
        vocab_vectors = torch.tensor(vocab_vectors)
        embeddings_vectors[(name_, size_)] = vocab_vectors
    
    print("Finished")
    return embeddings_vectors
