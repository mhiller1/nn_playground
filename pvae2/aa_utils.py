
import random

import torch
import torch.utils.data

aminoacids = "_DECNFTYQSMWVALIGPHKR"

def seq_to_tensor(seq):
    out = []
    for i in seq:
        out.append(aminoacids.index(i))
    return torch.nn.functional.one_hot(torch.tensor(out), len(aminoacids))

def tensor_to_seq(ten):
    a = ten.argmax(1)
    out = ""
    for i in a:
        out += aminoacids[i]
    return out

def random_seq(length):
    return "".join(random.choice(aminoacids[1:-1]) for i in range(length))

def pad_seq(input, length):
    for i in range(length - len(input)):
        input += "_"
    return input

class ProteinGenerator(torch.utils.data.Dataset):
    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        is_full = bool(random.getrandbits(1))
        if is_full:
            length = 512
        else:
            length = random.randrange(32, 512)
        ten = seq_to_tensor(pad_seq(random_seq(length), 512))
        return ten
