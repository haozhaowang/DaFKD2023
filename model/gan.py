import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: batch, width, height, channel=1
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x

def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
        
class GeneratorCifar(nn.Module):
    def __init__(self, noise_dimension):
        super(GeneratorCifar, self).__init__()
        self.n_channel = 3
        self.n_g_feature = 64
        self.module = nn.Sequential(
            nn.ConvTranspose2d(noise_dimension, 4 * self.n_g_feature, kernel_size=4, bias=False),
            nn.BatchNorm2d(4 * self.n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(4 * self.n_g_feature, 2 * self.n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * self.n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(2 * self.n_g_feature, self.n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(self.n_g_feature, self.n_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.module(x)
        return x


class DiscriminatorCifar(nn.Module):
    def __init__(self):
        super(DiscriminatorCifar, self).__init__()
        self.n_channel = 3
        self.n_d_feature = 64
        self.module = nn.Sequential(
            nn.Conv2d(self.n_channel, self.n_d_feature, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.n_d_feature, 2 * self.n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * self.n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2 * self.n_d_feature, 4 * self.n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * self.n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * self.n_d_feature, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.module(x)
        return x