import torch


SOS_TOKEN = 0
EOS_TOKEN = 1

MAX_LENGTH = 20

TRAIN_SRC_FNAME = (
    "trucCHS.txt"
)
TRAIN_TGT_FNAME = (
    "trucENU.txt"
)
DEV_SRC_FNAME = (
    "valCHS.txt"
)
DEV_TGT_FNAME = (
    "valENU.txt"
)
TEST_SRC_FNAME = (
    "testCHS.txt"
)
TEST_TGT_FNAME = (
    "testENU.txt"
)


ENCODER_FNAME = (
    "encoder.pt"
)
DECODER_FNAME = (
    "decoder.pt"
)
BASELINE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
