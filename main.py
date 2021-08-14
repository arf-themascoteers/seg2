import torch
import torchvision.io
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn


class Segnet(nn.Module):
    def __init__(self):
        super(Segnet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,6,(4,4)),
            nn.ReLU(),
            nn.Conv2d(6, 12, (4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 6, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, (3, 3)),
            nn.Sigmoid()
        )
        self.mask1 = nn.Parameter()
        self.mask2 = nn.Parameter()

    def forward(self, x):
        x = torch.cat((x,self.mask1, self.mask2), dim=1)
        return self.net(x)

image = Image.open("image.png")
mask = Image.open("mask.png")

transform = torchvision.transforms.ToTensor()
image = transform(image)
mask = transform(mask)
mask1 = mask
mask2 = mask1.clone()
mask2[mask2==0] = 2
mask2[mask2==1] = 0
mask2[mask2==2] = 1

s = Segnet()
s.train()
criterion = nn.MSELoss()


num_epochs = 10
for epoch in range(num_epochs):
    image = image.reshape(1, 1, 28, 28)
    mask1 = mask1.reshape(1, 1, 28, 28)
    mask2 = mask2.reshape(1, 1, 28, 28)
    s.mask1 = nn.Parameter(mask1)
    s.mask2 = nn.Parameter(mask2)
    s.mask1.requires_grad = False
    s.mask2.requires_grad = False
    optimizer = torch.optim.Adam(s.parameters(), lr=1e-3, weight_decay=1e-5)
    y_hat = s(image)
    y = torch.zeros_like(y_hat)
    loss = criterion(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')

torch.save(s.state_dict(), 'cnn.h5')

image = image.reshape(1, 1, 28, 28)
mask1 = torch.rand_like(image)
mask2 = torch.rand_like(image)
for epoch in range(num_epochs):
    plt.imshow(mask1.reshape(28,28))
    plt.show()
    plt.imshow(mask2.reshape(28,28))
    plt.show()
    s = Segnet()
    s.mask1 = nn.Parameter(mask1)
    s.mask2 = nn.Parameter(mask2)
    s.load_state_dict(torch.load("cnn.h5"))
    s.mask1 = nn.Parameter(mask1)
    s.mask2 = nn.Parameter(mask2)
    s.mask1.requires_grad = True
    s.mask2.requires_grad = True
    optimizer = torch.optim.Adam(s.parameters(), lr=1e-3, weight_decay=1e-5)
    y_hat = s(image)
    y = torch.zeros_like(y_hat)
    loss = criterion(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    mask1 = s.mask1.data
    mask2 = s.mask2.data

