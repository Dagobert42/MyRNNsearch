import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# first a simple encoder-decoder model was implemented
# to develop an understanding of the task at hand
# and obtain a baseline on NMT performance