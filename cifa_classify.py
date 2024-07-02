import torch
from torch import nn
from torchvision import transforms
import time
from Model import cdcd
from data_loader import Dataloader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, criterion, optimizer):
    model.train()
    start_time = time.time()
    loss_sum = 0.0
    avg_accuracy = 0.0
    for ind, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.long().to(device)
        output = model(images)
        # print(output)
        # print(labels)
        loss = criterion(output, labels)
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct = (output.argmax(dim=1) == labels).sum().item()
        accuracy = correct / len(labels)
        avg_accuracy += accuracy
        if ind % 100 == 0:
            print('-----------第{}次训练-----------'.format(ind))
            print('  Loss: {:.4f}    accuracy: {:.2f}'.format(loss.item(), accuracy))
            print('  Avg accuracy: {:.2f}'.format(avg_accuracy/(ind+1)))
    end_time = time.time()
    print('Train time: {} seconds'.format(end_time - start_time))
    return loss_sum, avg_accuracy / len(train_loader)


def test(model,test_loader,criterion):
    model.eval()
    start_time = time.time()
    loss_sum = 0.0
    avg_accuracy = 0.0
    with torch.no_grad():
        for ind, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.long().to(device)
            output = model(images)
            loss = criterion(output, labels)
            loss_sum += loss.item()
            correct = (output.argmax(dim=1) == labels).sum().item()
            accuracy = correct / len(labels)
            avg_accuracy += accuracy
            if ind % 100 == 0:
                print('---------------第{}次测试-----------'.format(ind))
                print('  Loss: {:.4f}  accuracy: {:.2f}'.format(loss.item(), accuracy))
                print('  Test Accuracy: {:.2f}'.format(avg_accuracy/(ind+1)))
    end_time = time.time()
    print('Test time: {} seconds'.format(end_time - start_time))
    return loss_sum, avg_accuracy / len(test_loader)


if __name__ == "__main__":
    model = cdcd().to(device)
    writer = SummaryWriter(log_dir='./logs')
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataloader = Dataloader(data_dir='./dataset/MSTAR', batch_size=8, num_workers=2, transform=transform, split=0.8)
    train_loader, test_loader = dataloader.get_loader()
    for epoch in range(10):
        print(f'Epoch:  {epoch} / 10')
        print('-----------开始训练------------')
        train_running_loss, train_avg_accuracy= train(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)
        torch.save(model.state_dict(), f'./model/cdcd_model_{epoch}.pth')
        print(f'Epoch: {epoch} Train Loss: {train_running_loss} Running Average Accuracy: {train_avg_accuracy}')
        writer.add_scalar('Train_Loss', train_running_loss, epoch)
        writer.add_scalar('Train_Accuracy', train_avg_accuracy, epoch)
        print(f'模型cdcd_model_{epoch}.pth 已保存')
        print('-----------开始测试------------')
        test_running_loss, test_avg_accuracy = test(model=model, test_loader=test_loader,criterion=criterion)
        print(f'Epoch: {epoch} Test Loss: {test_running_loss} Running Average Accuracy: {test_avg_accuracy}')
        writer.add_scalar('Test_Loss', test_running_loss, epoch)
        writer.add_scalar('Test_Accuracy', test_avg_accuracy, epoch)
    print('\n\n-----------------------训练结束---------------------------')
    writer.close()




