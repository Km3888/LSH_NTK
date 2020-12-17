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
from ntk_utils import compute_approximation,get_kl_div,get_grad_vector,get_param_vector
from models import Net,LM

import random

def train(args, model, device, train_loader, optimizer, epoch,writer,test_loader,ntk_dict):
    # sampling_method = "SGD"
    # sampling_method = "True_Adaptive_SGD"
    sampling_method = "Approx_Adaptive_SGD"
    
    model.train()
    train_loss=0

    train_data = [x for x in train_loader]
    train_size = len(train_data)

    num_unique_samples_this_epoch = 0
    unique_samples_this_epoch = set()

    batch_idx = 0
    print("iter\ttrain_loss\ttest_loss\tnum_unique_samples_this_epoch")
    while batch_idx < train_size:
        
        if sampling_method == "SGD":
            data, target = train_data[random.randint(0, train_size-1)]

        if sampling_method == "True_Adaptive_SGD":
            samples_with_magnitudes = []
            sum_of_magnitudes = 0
            for item in train_data:
                optimizer.zero_grad()
                output = model(item[0].to(device))
                loss = F.mse_loss(output, item[1].to(device))
                loss.backward()
                grads = []
                for param in model.parameters():
                    grads.append(param.grad.view(-1))
                magnitude=torch.norm(torch.cat(grads))
                samples_with_magnitudes.append((magnitude, item))
                sum_of_magnitudes += magnitude
            # weights proportional to gradient magnitude, all weights add to 1
            samples_with_normalized_magnitudes = [(x[0]/sum_of_magnitudes, x[1]) for x in samples_with_magnitudes]
            rand = random.random()
            for x in samples_with_normalized_magnitudes:
                rand -= x[0]
                if rand < 0:
                    data, target = x[1]
                    break

        if sampling_method == "Approx_Adaptive_SGD":
            if batch_idx % 50 == 0: # Only do this once every epoch, can adjust to re-do every % x iterations
                y_0 = [model(x[0]).view(-1) for x in train_data]
                w_0=get_param_vector(model)
                grad_vecs=[]
                for item in train_data:
                    optimizer.zero_grad()
                    output = model(item[0].to(device))
                    loss = F.mse_loss(output, item[1].to(device))
                    loss.backward()
                    grads = []
                    for param in model.parameters():
                        grads.append(param.grad.view(-1))
                    grad_vecs.append(torch.cat(grads))
                G = torch.stack(grad_vecs)

            samples_with_estimated_magnitudes = []
            sum_of_estimated_magnitudes = 0

            w = get_param_vector(model)
            for i, item in enumerate(train_data):
                est_magnitude = torch.zeros([1])
                norm_G_i = torch.norm(G[i]).view(-1)
                est_magnitude += (torch.dot(w, norm_G_i * G[i]))
                est_magnitude -= (norm_G_i * (item[1] - y_0[1] + torch.dot(G[i], w_0)))
                est_magnitude = float(abs(est_magnitude))

                samples_with_estimated_magnitudes.append((est_magnitude, item))
                sum_of_estimated_magnitudes += est_magnitude
            
            samples_with_normalized_estimated_magnitudes = [(x[0]/sum_of_estimated_magnitudes, x[1]) for x in samples_with_estimated_magnitudes]
            rand = random.random()
            # print(sum([x[0] for x in samples_with_normalized_estimated_magnitudes]))
            # print(sum_of_estimated_magnitudes, rand, [round(x[0], 3) for x in samples_with_normalized_estimated_magnitudes][:10])
            for i, x in enumerate(samples_with_normalized_estimated_magnitudes):
                rand -= x[0]
                if rand < 0:
                    data, target = x[1]
                    break

        
        data, target = data.to(device), target.to(device)

        if data not in unique_samples_this_epoch:
            unique_samples_this_epoch.add(data)
            num_unique_samples_this_epoch += 1

        #first_data=data[0].view([1,90])
        #grad_vector = get_grad_vector(model,first_data)
        optimizer.zero_grad()
        output = model(data).view(-1)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss+=loss
        if batch_idx % args.log_interval == 0:
            curr_iter= (epoch-1)*len(train_loader)+batch_idx

            test_loss=test(model,device,test_loader,num_samples=5000)
            train_loss=calc_train_loss(model,device,train_loader,num_samples=5000)

            print(str(curr_iter) + "\t" + str(round(train_loss, 4)) + "\t\t" + str(round(test_loss, 4)) + "\t\t" + str(num_unique_samples_this_epoch))

            # writer.add_scalar('test loss',test_loss,curr_iter)
            # writer.add_scalar('train loss',train_loss,curr_iter)

            train_loss = 0

            # ntk_kl_div,uniform_kl_div = get_kl_div(model,ntk_dict)
            # writer.add_scalar('ntk kl div',ntk_kl_div,curr_iter)
            # writer.add_scalar('random choice kl div',uniform_kl_div,curr_iter)
            if args.dry_run:
                break
        batch_idx += 1
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
    parser.add_argument('--log-interval', type=int, default=25, metavar='N',
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

    dataset1,dataset2=MSDDataset(train=True,size=1000),MSDDataset(train=False)

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
