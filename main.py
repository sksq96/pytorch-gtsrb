from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import os
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--experiment', type=str, default='test', metavar='D',
                    help="tensorboard experiment")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms, train_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=train_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = AdamW.AdamW(model.parameters(), lr=args.lr)


from tensorboard import TensorBoard
runs_dir = f"runs/{args.experiment}/{time.asctime(time.localtime())}/"
tb = TensorBoard(runs_dir)
step = 0

tick = time.time()

def train(epoch):
    global step
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        step += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            correct, accuracy = validate_batch(train_loader)
            vcorrect, vaccuracy = validate_batch(val_loader)
            tock = (time.time() - tick)/60
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.1f}%)\tVAcc: {}/{} ({:.1f}%)\tTimes: {:.1f} min'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                correct, args.batch_size, accuracy,
                vcorrect, args.batch_size, vaccuracy, tock)
            )
            tb.scalar_summary("controller/loss", loss.item(), step)
            tb.scalar_summary("controller/accuracy", accuracy, step)
            tb.scalar_summary("controller/vaccuracy", vaccuracy, step)


def validate_batch(loader):
    model.eval()
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    output = model(data)
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()
    accuracy = 100.0 * correct / len(data)
    model.train()
    return correct, accuracy

def validation():
    global step
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100.0 * correct / len(val_loader.dataset)))
    tb.scalar_summary("validation/accuracy", 100. * correct / len(val_loader.dataset), step)
    tb.scalar_summary("validation/loss", validation_loss, step)


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()

    directory = f'save/{args.experiment}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    model_file = f'{directory}/model_{epoch}.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')


