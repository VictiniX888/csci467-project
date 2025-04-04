import torch

SOS_token = 0
EOS_token = 1

BASELINE = True

MAX_LENGTH = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
