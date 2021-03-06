import torch
from torch import nn
import torchvision
from torchvision.models.mobilenet import mobilenet_v2
import json 
import numpy as np
import bcolz 
import pickle
from utils import create_word_embedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, encoder_out, decoder_hidden):

        att1 = self.encoder_att(encoder_out)  
        att2 = self.decoder_att(decoder_hidden)  
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  
        alpha = self.softmax(att)  
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  

        self.embedding = create_word_embedding() 
        self.dropout = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True) 
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        # self.f_beta = nn.Linear(decoder_dim, encoder_dim)  
        self.attention_learner_1 = nn.Linear(decoder_dim, 512)
        self.attention_learner_2 = nn.Linear(512, encoder_dim)
        self.relu = nn.LeakyReLU(0.01) 
        self.sigmoid = nn.Sigmoid()
        self.fc_1 = nn.Linear(decoder_dim, 512)
        self.fc_2 = nn.Linear(512, vocab_size)  
        self.init_weights()  

    def init_weights(self):

        self.fc_1.bias.data.fill_(0)
        self.fc_2.bias.data.fill_(0)
        self.fc_1.weight.data.uniform_(-0.1, 0.1)
        self.fc_2.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

   
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)  

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  

        
        decode_lengths = (caption_lengths - 1).tolist()

        
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            attention_intermediate_representation = self.relu(self.attention_learner_1(h[:batch_size_t]))
            gate = self.sigmoid(self.attention_learner_2(attention_intermediate_representation))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  
            preds = self.fc_2(self.tanh(self.fc_1(self.dropout(h))))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
