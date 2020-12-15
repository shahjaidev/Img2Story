import torch
from torch import nn
import torchvision
from torchvision.models.mobilenet import mobilenet_v2
import json 
import numpy as np
import bcolz 
import pickle

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_word_embedding():
    # using word2vec
    vectors = bcolz.open(f'./Glove/6B.300.dat')[:]
    words = pickle.load(open(f'./Glove/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'./Glove/6B.300_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    with open('./caption_data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json', 'r') as j:
        target_vocab = json.load(j)
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0

    for i, word in enumerate(target_vocab):
        word = word.lower()
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))

    emb_layer = nn.Embedding(len(target_vocab),300)
    weights_matrix = torch.tensor(weights_matrix)
    emb_layer.load_state_dict({'weight': weights_matrix})

    return emb_layer


def accuracy(scores, targets, k):

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  
    return correct_total.item() * (100.0 / batch_size)
