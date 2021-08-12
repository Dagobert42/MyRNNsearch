import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################_ENCODER_######################

class Encoder(nn.Module):
    pass
######################_ATTENTION_######################

class Attention(nn.Module):
    pass

######################_DECODER_######################

class Decoder(nn.Module):
    pass

######################_RNNsearch_######################

class RNNsearch(nn.Module):

    def beam_search(self, decoder_out, k):
        pass
