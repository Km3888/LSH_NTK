from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import MSDDataset
# import pydevd_pycharm
from torch.utils.tensorboard import SummaryWriter
# pydevd_pycharm.settrace('km3888@hegde-lambda-1.engineering.nyu.edu', port=22, stdoutToServer=True, stderrToServer=True)
from ntk_utils import compute_approximation,get_kl_div
from models import Net,LM

def train(args, model, device, train_loader, optimizer, epoch,writer,test_loader,ntk_dict):
    model.train()
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #first_data=data[0].view([1,90])
        #grad_vector = get_grad_vector(model,first_data)
        optimizer.zero_grad()
        output = model(data).view(-1)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss+=loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            curr_iter= (epoch-1)*len(train_loader)+batch_idx

            test_loss=test(model,device,test_loader,num_samples=5000)
            train_loss=calc_train_loss(model,device,train_loader,num_samples=5000)

            writer.add_scalar('test loss',test_loss,curr_iter)
            writer.add_scalar('train loss',train_loss,curr_iter)

            train_loss = 0

            ntk_kl_div,uniform_kl_div = get_kl_div(model,ntk_dict)
            writer.add_scalar('ntk kl div',ntk_kl_div,curr_iter)
            writer.add_scalar('random choice kl div',uniform_kl_div,curr_iter)
            if args.dry_run:
                break
    return train_loss

'''Estimates the training loss by taking specified number of samples and averaging loss over them'''
def calc_train_loss(model, device, train_loader,num_samples=None):
    model.eval()
    train_loss = 0
    if num_samples is None:
        num_samples=len(train_loader.dataset)
    with torch.no_grad():
        for i,(data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data).view(-1)
            indv_loss = F.mse_loss(output, target, reduction='sum').item()
            train_loss += indv_loss  # sum up batch loss
            if i>num_samples:
                break

    train_loss /= num_samples

    return train_loss


'''Estimates the validation loss by taking specified number of samples and averaging loss over them'''
def test(model, device, test_loader,num_samples=None):
    model.eval()
    test_loss = 0
    if num_samples is None:
        num_samples = len(test_loader.dataset)
    total_samples=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).view(-1)
            indv_loss = F.mse_loss(output, target, reduction='sum').item()
            test_loss += indv_loss  # sum up batch loss
            total_samples+=data.shape[0]
            if total_samples>num_samples:
                break
    test_loss /= total_samples
    return test_loss

def estimate_output(y_0, w, w0, grad):
    diff = w - w0
    return y_0 + torch.dot(grad, diff)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--train_size', type=int, default=463715)
    parser.add_argument('--test_size', type=int, default=1000)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    writer=SummaryWriter()

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,#TODO change this back to 1
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1,dataset2=MSDDataset(train=True),MSDDataset(train=False)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    '''NTK dataset is used to evaluate the KL divergence between sampling distribution given by true gradient
    magnitudes and sampling distribution given by the NTK estimate of the magnitudes. Using the full dataset for this
    would be costly so we just use a subset to evaluate on.'''
    ntk_dataset = MSDDataset(train=True,size=100)
    ntk_loader = torch.utils.data.DataLoader(ntk_dataset,batch_size=100)
    ntk_data,ntk_target= next(iter(ntk_loader))
    ntk_data,ntk_target=ntk_data.to(device),ntk_target.to(device)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2,**test_kwargs)

    y_0,w_0,G=compute_approximation(model,ntk_data)
    ntk_dict={'y_0':y_0,'w_0':w_0,'G':G,'data':ntk_data,'targets':ntk_target}

    for epoch in range(1, args.epochs + 1):
        train_loss=train(args, model, device, train_loader, optimizer, epoch,writer,test_loader,ntk_dict)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "MSD_model.pt")



if __name__ == '__main__':
    main()
