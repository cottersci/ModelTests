'''
    Functions to train the nerual network and to track training progress.

    Uses tensorboardX package for logging.

    python Train.py --help
'''
import math
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

from torchvision.utils import make_grid
import scipy.stats

from pytorch_utils import vis, init, util, batcher

from tensorboardX import SummaryWriter

import conv
import utils
##
### ---  VARIABLE DEFINITIONS -------------------------------- ####
##

#Constants
EPS = 1e-8 #Small value added to avoid undefined calculatoins e.g.: log(0 + EPS)

#Parser
parser = init.parser()
parser.add_argument('z_dim', type=int, help='Z dimension')
parser.add_argument('--download',action='store_true',help='Download Dataset')
parser.add_argument('--Net',
                    default="conv",
                    help='Linear (linear) or Convolutional (conv) network.')
parser.add_argument('--dataset',
                    default="MNIST",
                    help='choices: MNIST, SVHN. default=MNIST')
parser.add_argument('--dataset-loc',
                    default=".",
                    help='Location of/to download dataset. default=./')
parser.add_argument('--lr',
                    type=float,
                    default=1e-5,
                    help='Learning rate for D and G networks, default=1e-4')
parser.add_argument('--Nimages',
                    type=float,
                    default=2,
                    help='''The test phase will produce a grid
                                of Nimages^z_dim images''')
parser.add_argument('--noise-dim',
                    type=int,
                    default=16,
                    help='''Number of noise dimensions''')
opt = parser.parse_args()
print(opt)

# Dataset
if(opt.dataset == 'MNIST'):
    train_loader = init.mnist(opt.batch_size,
                              opt.dataset_loc,
                              train=True,
                              download=opt.download,
                              pin_memory=opt.no_cuda)
elif(opt.dataset == "SVHN"):
    train_loader = init.svhn(opt.batch_size,
                              opt.dataset_loc,
                              train=True,
                              download=opt.download,
                              pin_memory=opt.no_cuda)
else:
    print('Invalid --dataset choice')
    exit()

# Network arch
Gx = conv.Gx(opt.z_dim,opt.noise_dim)
Gz = conv.Gz(opt.z_dim,opt.noise_dim)
D = conv.D(opt.z_dim)

# Training Variables
opto_Gx = optim.Adam(Gx.parameters(), lr=opt.lr) # Training
opto_Gz = optim.Adam(Gz.parameters(), lr=opt.lr) # Training
opto_D = optim.Adam(D.parameters(), lr=opt.lr) # Training

log = SummaryWriter(opt.logdir,opt.comment) # Logging
batch = batcher() # Logging

gauss = torch.FloatTensor(opt.batch_size,opt.z_dim) # z ~ p(z)
noise = torch.FloatTensor(opt.batch_size,opt.z_dim) # ~ U()
##
### ---  FUNCTION DEFINITIONS -------------------------------- ####
##

def train(epoch):
    D.train()
    Gx.train()
    Gz.train()
    '''
        Do 1 epoch of training, where an epoch is 1 iteration though the entire
        dataset.

        :param epoch (int): number of epochs trained, used for logging
    '''
    for batch_idx in range(len(train_loader)):
        batch.batch()

        ####
        # Train D
        opto_D.zero_grad()

        # Prepare batch
        img0, _ = next(train_loader.__iter__())
        if opt.no_cuda:
            img0 = img0.cuda()
        x_real  = Variable(img0)
        gauss.resize_(img0.size()[0],opt.z_dim)
        noise.resize_(img0.size()[0],opt.noise_dim)

        # p_q = P(x_real,z_gen)
        noise.uniform_()
        Gz_z, _, _ = Gz(x_real,noise)
        p_q, _ = D(x_real,Gz_z)

        # p_p = P(x_gen,z), z ~ N(0,1)
        gauss.normal_()
        noise.uniform_()
        z = Variable(gauss)
        Gx_x = Gx(z,noise)
        p_p, D_z = D(Gx_x,z)

        #D loss
        opto_D.zero_grad()
        Loss_d = -torch.log(p_q + EPS).mean() - torch.log(1 - p_p + EPS).mean()
        Loss_MId = (D_z - z).pow(2).sum(dim=1).mean()
        Loss_d.backward(retain_graph=True)
        Loss_MId.backward(retain_graph=True)
        opto_D.step()
        batch.add('loss/D',Loss_d.item())
        batch.add('loss/MI',Loss_MId.item())
        batch.add('gradients/D',utils.grad_norm(D.parameters()))

        ####
        # Train G
        opto_Gz.zero_grad()
        opto_Gx.zero_grad()
        Loss_gz = -torch.log(1 - p_q + EPS).mean() - torch.log(p_p + EPS).mean()
        Loss_MIgz = (D_z - z).pow(2).sum(dim=1).mean()
        Loss_gz.backward(retain_graph=True)
        opto_Gz.step()
        Loss_MIgz.backward()
        opto_Gx.step()
        batch.add('loss/Gxz',Loss_gz.item())
        batch.add('gradients/Gz',utils.grad_norm(Gx.parameters()))
        batch.add('gradients/Gx',utils.grad_norm(Gz.parameters()))

        ##
        # Progress Reporting
        if batch_idx % 125 == 1:
            print('Epoch: %d [%d/%d]: ' %
                   (
                     epoch,
                     batch_idx * len(img0),
                     len(train_loader.dataset),
                   ),
                  end = '')
            batch.report()
            print('',flush=True)

            #Universal step is total number of batches trained on
            batch.write(log,epoch * len(train_loader) + batch_idx)

def test(epoch):
    '''
        Gerneate some logging information that is too computationally
        intensive to log on each batch.

        :param epoch (int): number of epochs trained, used for logging
    '''

    #Generate a grid of points between approx. -3 and 3, where the distance
    #between points is inversely proportional to the PDF of a normal distirbution.
    L = 0.00135 + 0.99865 # CDF percentages from a N(0,1) coorespoding to -3 and 3
    points = np.arange(1,opt.Nimages+1) * L / (opt.Nimages + 1)
    points = np.tile(points,(opt.z_dim,1))
    mesh = np.meshgrid(*points)
    z = np.concatenate(mesh).reshape(opt.z_dim,-1).T
    z = scipy.stats.norm.ppf(z)

    z = torch.from_numpy(z).contiguous().float()
    if opt.no_cuda:
        z = z.cuda()

    z = Variable(z)
    noise.resize_(z.size()[0],opt.noise_dim)
    noise.uniform_()
    x = Gx.eval()(z,noise)

    x = x.data
    z = z.data

    if opt.no_cuda:
        x = x.cpu()
        z = z.cpu()

    ##
    # Generate Image
    buf = make_grid(x, padding=3,nrow=math.floor(math.sqrt(opt.Nimages ** opt.z_dim)),range=(0,1))
    log.add_image('Generated', buf, (epoch + 1) * len(train_loader))

    ##
    # Project the esimated z values
    img0, target = next(train_loader.__iter__())
    noise.resize_(img0.size()[0],opt.noise_dim).normal_()

    noise.uniform_() # noise ~ N(0,1)
    if opt.no_cuda:
        _, z_gen, _ = Gz(Variable(img0.cuda()),noise)
        z_gen = z_gen.data.cpu()
    else:
        z_gen = Gz(Variable(img0))
        z_gen = z_gen.data

    if(opt.z_dim == 2):
        buf = vis.scatterBWImages(z_gen.numpy(),img0.numpy())
        log.add_image('Real', buf, (epoch + 1) * len(train_loader))
    else:
        log.add_embedding(z_gen,
                         metadata=target,
                         label_img=img0,
                         global_step=(epoch + 1) * len(train_loader),
                         tag='%d Epoch' % epoch)
##
### ---  TRAINING ------------------------------------------------ ####
##
if(opt.no_cuda):
    Gx = Gx.cuda()
    Gz = Gz.cuda()
    D = D.cuda()
    gauss = gauss.cuda()
    noise = noise.cuda()

for epoch in range(int(opt.epochs)):
    train(epoch)
    test(epoch)

    if opt.save_every != 0 and epoch % opt.save_every == 0:
        folder_name = './CHECKPOINT-%d' % (epoch)
        util.save_nets(folder_name,
                        {
                         'D': D,
                         'Gx': Gx,
                         'Gz' : Gz
                        })

folder_name = './CHECKPOINT-END'
util.save_nets(folder_name,
                {
                 'D': D,
                 'Gx': G,
                 'Gz' : Gx
                })
