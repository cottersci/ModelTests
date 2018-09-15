'''
    Functions to train the nerual network and to track training progress.

    Uses tensorboardX package for logging.

    python Train.py --help
'''
import math
import itertools
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
opt = utils.parse_args()
print(opt)

# Dataset
train_loader = utils.init_dataset_loader(opt.dataset,opt.batch_size,opt.data_loc,pin_memory=opt.no_cuda)

# Network arch
Gx = conv.Gx(opt.z_dim,opt.noise_dim)
Gz = conv.Gz(opt.z_dim,opt.noise_dim)
D = conv.D(opt.z_dim)

# Training Variables
betas = (0.5,0.999)
opto_Gx = optim.Adam(Gx.parameters(), lr=opt.lr, betas=betas) # Training
opto_Gz = optim.Adam(Gz.parameters(), lr=opt.lr, betas=betas)
opto_D = optim.Adam(D.parameters(), lr=opt.lr, betas=betas) # Training

log = SummaryWriter(opt.logdir,opt.comment) # Logging
batch = batcher() # Logging

# Reused variables
noise = torch.FloatTensor(opt.batch_size,opt.noise_dim)
gauss = torch.FloatTensor(opt.batch_size,opt.z_dim) # z ~ p(z)

##
### ---  FUNCTION DEFINITIONS -------------------------------- ####
##
def train(epoch):
    '''
        Do 1 epoch of training, where an epoch is 1 iteration though the entire
        dataset.

        Args:
            epoch (int): number of epochs trained, used for logging
            noise (:class:`FloatTensor`): vector to hold the noise distribution
    '''
    D.train()
    Gx.train()
    Gz.train()
    for batch_idx in range(len(train_loader)):
        batch.batch()

        ####
        # Prepare batch
        img0, _ = next(train_loader.__iter__())
        if opt.no_cuda:
            img0 = img0.cuda()
        x_real  = Variable(img0)
        noise.resize_(x_real.size()[0],opt.noise_dim)
        gauss.resize_(img0.size()[0],opt.z_dim)

        ####
        # Train

        # p_q = P(x_real,z_gen)
        noise.normal_() # noise ~ N(0,1)
        z_gen = Gz(x_real,Variable(noise))
        p_q = D(x_real,z_gen)

        # p_p = P(x_gen,z), z ~ N(0,1)
        gauss.normal_() # z ~ N(0,1)
        noise.normal_()  # noise ~ N(0,1)
        z = Variable(gauss)
        x_gen = Gx(z,Variable(noise))
        p_p = D(x_gen,z)

        # D loss
        opto_D.zero_grad()
        Loss_d = -torch.log(p_q + EPS).mean() - torch.log(1 - p_p + EPS).mean()
        Loss_d.backward(retain_graph=True)
        batch.add('gradients/D',utils.grad_norm(D.parameters()))
        batch.add('loss/d',Loss_d.item())
        opto_D.step()

        # I(x_hat,z)
        # opto_Gx.zero_grad()
        # opto_D.zero_grad()
        # Loss_I = (z_est - z + EPS).pow(2).sqrt().sum(dim=1).mean()
        # Loss_I.backward(retain_graph=True)
        # opto_D.step()
        # opto_Gx.step()
        # batch.add('gradients/I_D',utils.grad_norm(D.parameters()))
        # batch.add('gradients/I_Gx',utils.grad_norm(Gx.parameters()))
        # batch.add('loss/I',Loss_I.item())

        # Gx,Gz
        opto_Gx.zero_grad()
        opto_Gz.zero_grad()
        Loss_g = -torch.log(1 - p_q + EPS).mean() - torch.log(p_p + EPS).mean()
        Loss_g.backward()
        opto_Gx.step()
        opto_Gz.step()
        batch.add('loss/g',Loss_g.item())
        batch.add('gradients/Gx',utils.grad_norm(Gx.parameters()))
        batch.add('gradients/Gz',utils.grad_norm(Gz.parameters()))

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
        Generate some logging information that is too computationally
        intensive to log on each batch.

        Args:
            param epoch (int): number of epochs trained, used for logging
            noise (:class:`FloatTensor`): vector to hold the noise distribution
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
    noise.resize_(z.size()[0],opt.noise_dim).normal_()
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

    noise.normal_() # noise ~ N(0,1)
    if opt.no_cuda:
        z_gen = Gz(Variable(img0.cuda()),Variable(noise))
        z_gen = z_gen.data.cpu()
    else:
        z_gen = Gz(Variable(img0),Variable(noise))
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
    G = Gx.cuda()
    D = D.cuda()
    Gz = Gz.cuda()
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
                         'Gz': Gz
                        })

folder_name = './CHECKPOINT-END'
util.save_nets(folder_name,
                {
                 'D': D,
                 'Gx': Gx,
                 'Gz': Gz
                })
