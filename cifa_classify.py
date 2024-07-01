import torch
import torchvision
from torch.nn import Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

Cifar_train = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(Cifar_train, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter('log')
writer2 = SummaryWriter('log_seq')


class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=1),  # 3x32x32 -> 32x32x32
            nn.MaxPool2d(2),  # 33x16x16
            nn.Conv2d(32, 64, 3, padding=1, stride=1),  # 64x16x16
            nn.MaxPool2d(2),  # 64x8x8
            Flatten(),  # 4096
            Linear(64 * 8 * 8, 64),  # 64
            Linear(64, 10)  # 10
        )

    def forward(self, x):
        x = self.block1(x)
        return x


model = cnn_model()
print(model)
loss = nn.CrossEntropyLoss()
step = 1
epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
ins = 1
for epoch in range(epochs):
    sum_loss = 0.0
    for data in data_loader:
        img, label = data
        if ins == 1:
            writer2.add_graph(model, img)
            ins -=1
        output = model(img)
        result_loss = loss(output, label)
        sum_loss += result_loss
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        # print(output.shape)
        if step % 100 == 0:
            print(step)
        step += 1
    writer.add_scalar('loss', sum_loss, epoch)
    print(sum_loss)
    print(epoch)
writer.close()
