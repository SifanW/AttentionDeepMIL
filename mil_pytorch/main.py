from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from dataloader import AngioBags
from model import Attention

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--val_step',type=int, default=3, metavar='V',
                    help='number of epochs to valid (default: 2)')
parser.add_argument('--lr', type=float, default=1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-2, metavar='R',
                    help='weight decay(default: 10e-5)')
parser.add_argument('--patch_length', type=int, default=128, metavar='T',
                    help='frame divided into patches with patch_length edge')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('GPU is ON!')

print('Loading Train, Validation and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(AngioBags(seed=args.seed,
                                               _is_train=True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

val_loader = data_utils.DataLoader(AngioBags(seed=args.seed,
                                             _is_val=True),
                                   batch_size=1,
                                   shuffle=True,
                                   **loader_kwargs)

test_loader = data_utils.DataLoader(AngioBags(seed=args.seed,
                                              _is_test=True),
                                    batch_size=1,
                                    shuffle=True,
                                    **loader_kwargs)


print("*"*10 + ' Init Model '+ "*"*10)
model = Attention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def _convert_2_tensor(imgs):
    #print(imgs.shape)
#     imgs = imgs.reshape(imgs.shape[0],imgs.shape[1] * imgs.shape[2]*imgs.shape[3])
#     print(imgs.shape)
#     imgs = np.asarray(imgs)
#     print(imgs.shape)
    #imgs = torch.from_numpy(imgs)
    print(imgs.shape)
    return imgs

def train(epoch):
    print("*"*10 + ' Start Training ' + "*"*10)
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        #data = torch.Tensor(data)
        data = data.float()
        bag_label = np.asarray(label)
        bag_label = torch.Tensor(bag_label)
        bag_label = bag_label.float()
        #print("data shape:",data.shape)
        #print("bag_label shape:",bag_label.shape)
        #data = _convert_2_tensor(data)

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
            
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data[0]
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))

def valid():
    print("-"*10 + '  validation '+ "-"*10)
    model.eval()
    val_loss = 0.
    val_error = 0.
    result = []
    for batch_idx, (data, label) in enumerate(val_loader):
        data = data.float()
        bag_label = np.asarray(label)
        bag_label = torch.Tensor(bag_label)
        bag_label = bag_label.float()

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        val_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        val_error += error
        bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
        result.append(bag_level)

    print('\nTrue Bag Label, Predicted Bag Label: {}\n'.format(result))
    val_error /= len(val_loader)
    val_loss /= len(val_loader)

    print('\nValid Set, Loss: {:.4f}, valid error: {:.4f}'.format(val_loss.cpu().numpy()[0], val_error))



def test():
    print("*"*10 + ' Start Testing ' + "*"*10)
    model.eval()
    test_loss = 0.
    test_error = 0.
    result = []
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.float()
        bag_label = np.asarray(label)
        bag_label = torch.Tensor(bag_label)
        bag_label = bag_label.float()

        
#         bag_label = label[0]
#        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error
        bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
        result.append(bag_level)

        if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            #bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))            
            #instance_level = list(zip(instance_labels.numpy()[0]),np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

#             print('\nTrue Bag Label, Predicted Bag Label: {}\n'
#                   'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))
            print('\nTrue Bag Label, Predicted Bag Label: {}\n'.format(bag_level))
            
#             print("data shape:",data.shape)
#             print("bag_label shape:",bag_label.shape)
#             print("bag_label:",bag_label)
#             print("test_loss:",test_loss)
#             print("predicted_label:",predicted_label)
# #             print("instance_labels:",instance_labels)
#             print("test_error:",test_error)
#             print("bag_level:",bag_level)
#             print("instance_level:",instance_level)
    print('\nTrue Bag Label, Predicted Bag Label: {}\n'.format(result))
    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))

if __name__ == "__main__":
    print(model)
    for step in range(5):
        for epoch in range(1, args.epochs + 1):
            train(step*20 + epoch)
            if epoch % args.val_step == 0:
                valid() 
        test()

