
import torch

class PVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, 3, padding=1, stride=2, bias=False), # 512*21
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv1d(4, 4, 3, padding=1, stride=2, bias=False), # 256*21
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv1d(4, 4, 3, padding=1, stride=2, bias=False), # 128*21 -> 64*21
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(4*64*21, 4*64*21),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(4*64*21, 2*64*21),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.li_mu = torch.nn.Linear(2*64*21, 2*64*21)
        self.li_log = torch.nn.Linear(2*64*21, 2*64*21)

        self.dec1 = torch.nn.Sequential(
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(2*64*21, 2*64*21),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(2*64*21, 4*64*21),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(4*64*21, 4*64*21),
        )
        self.dec2 = torch.nn.Sequential(
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.ConvTranspose1d(4, 4, 4, padding=1, stride=2, bias=False),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.ConvTranspose1d(4, 4, 4, padding=1, stride=2, bias=False),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.ConvTranspose1d(4, 1, 4, padding=1, stride=2, bias=False),
            torch.nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.enc1(x)
        x = x.view([-1, 4 * 64 * 21])
        x = self.enc2(x)
        return self.li_mu(x), self.li_log(x)

    def decode(self, x):
        x = self.dec1(x)
        x = x.view([-1, 4, 64 * 21])
        x = self.dec2(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        # print(f"x: {x.size()}, y: {z.size()}, z: {out.size()}")
        return out, mu, logvar
