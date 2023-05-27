import os
import time
import random
import argparse

import torch
import torch.utils.data
from accelerate import Accelerator

import aa_utils
import aa_model


# use_cuda = True
#
# if use_cuda:
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Protein VAE multitool')
parser.add_argument('-l', '--load', type=str, required=False, help='Path to load checkpoint')
parser.add_argument('-s', '--save', type=str, required=False, help='Path to save checkpoint')
parser.add_argument('-k', '--test', type=int, default=0, help='Test VAE with random data')
parser.add_argument('-t', '--train', type=int, default=0, help='Training epochs count')
parser.add_argument('-b', '--batch', type=int, default=1000, help='Training batch size')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Training learning rate')
args, _ = parser.parse_known_args()

random.seed(time.time() + os.getpid())

model = aa_model.PVAE()
if args.load != None:
    model.load_state_dict(torch.load(args.load))

accelerator = Accelerator()
device = accelerator.device
model.to(device)

pgen = aa_utils.ProteinGenerator()

if args.train > 0:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    data = torch.utils.data.DataLoader(pgen, batch_size=args.batch)
    model, optimizer, data = accelerator.prepare(model, optimizer, data)
    iteration = 0

    for i in range(args.train):
        iteration = 0
        model.train()
        for x in data:
            iteration += 1
            x = x.view([-1, 1, 512*21]).to(device=device, dtype=torch.float)
            z, mu, logvar = model.forward(x)
            bc_loss = torch.nn.functional.binary_cross_entropy(z, x, reduction="sum")
            ms_loss = torch.nn.functional.mse_loss(z, x, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = bc_loss + ms_loss + kl_loss
            optimizer.zero_grad()
            #loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            print(f"[{i+1}, {iteration}] \tloss: {loss/x.data.size(0)} \t(bce: {bc_loss/x.data.size(0)} \tmse: {ms_loss/x.data.size(0)} \tkl: {kl_loss/x.data.size(0)})")
        model.eval()
else:
    model.eval()

if args.test > 0:
    for i in range(args.test):
        seq = aa_utils.pad_seq(aa_utils.random_seq(random.randrange(128, 512)), 512)
        ten = aa_utils.seq_to_tensor(seq).view([1, 512*21]).to(device=device, dtype=torch.float)
        rec, _, _ = model.forward(ten)
        rec = aa_utils.tensor_to_seq(rec.to("cpu").view([512, 21]))
        print(f"{seq}\n{rec}\n\n")

if args.save != None:
    model.to("cpu")
    torch.save(model.state_dict(), args.save)
