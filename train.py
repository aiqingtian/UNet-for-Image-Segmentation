from os.path import join
from optparse import OptionParser
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch import optim
import matplotlib.pyplot as plt
from model import UNet
from dataloader import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_net(net,
              epochs=20,
              data_dir='data/cells/',
              n_classes=2,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              gpu=False):
    loader = DataLoader(data_dir)
    # N_train = loader.n_train()
    optimizer = optim.SGD(net.parameters(),
                            lr=lr,
                            momentum=0.99,
                            weight_decay=0.0005)

    # optimizer = optim.Adam(net.parameters())
    epochs_loss = 10
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')
        epoch_loss = 0
        for i, (img, label) in enumerate(loader):
            shape = img.shape
            label = label - 1
            # Create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)
            img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
            # Load image tensor to gpu
            if gpu:
                img_torch = img_torch.cuda()
            # Get prediction and getLoss()
            net = net.to(device)
            pred = net(img_torch)
            loss = getLoss(pred, label)
            epoch_loss += loss.item()
            # optimize weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        current_epochs_loss = epoch_loss/i
        if(current_epochs_loss < epochs_loss):
            torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/BestModelCP%d.pth' % (current_epochs_loss))
            epochs_loss = current_epochs_loss
            print('Best model saved')
        if(epoch%20 == 0):
            torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/SGDCP%d.pth' % (epoch + 1))
            print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))
    end = time.time()
    print('time cost is: ', end - start)
    # Displays test images with original and predicted masks after training
    loader.setMode('test')
    net.eval()
    with torch.no_grad():
        for _, (img, label) in enumerate(loader):
            shape = img.shape
            img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
            if gpu:
                img_torch = img_torch.cuda()
            pred = net(img_torch)
            pred_sm = softmax(pred)
            _,pred_label = torch.max(pred_sm,1)
            plt.subplot(1, 3, 1)
            plt.imshow(img*255.)
            plt.subplot(1, 3, 2)
            plt.imshow((label-1)*255.)
            plt.subplot(1, 3, 3)
            plt.imshow(pred_label.cpu().detach().numpy().squeeze()*255.)
            plt.show()

def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    return cross_entropy(p, target_label)

def softmax(input):
    # Implement softmax function
    p = input
    sum = torch.exp(input[0,0,:,:])+torch.exp(input[0,1,:,:])
    p[0,0,:,:] = torch.div(torch.exp(input[0,0,:,:]), sum)
    p[0,1,:,:] = torch.div(torch.exp(input[0,1,:,:]), sum)
    return p

def cross_entropy(input, targets):
    # Implement cross entropy
    total_samples = input.size()[2]*input.size()[3]
    pred_labels = choose(input, targets)
    ce = -torch.sum(torch.log(pred_labels))/total_samples
    return ce

def choose(pred_label, true_labels):
    size = pred_label.size()
    ind = np.empty([size[2]*size[3],3], dtype=int)
    i = 0
    for x in range(size[2]):
        for y in range(size[3]):
            ind[i,:] = [true_labels[x,y], x, y]
            i += 1
    pred = pred_label[0,ind[:,0],ind[:,1],ind[:,2]].view(size[2],size[3])
    return pred
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=50, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/cells/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    start = time.time()
    print('start time counting')
    args = get_args()
    net = UNet(n_classes=args.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        n_classes=args.n_classes,
        gpu=args.gpu,
        data_dir=args.data_dir)
