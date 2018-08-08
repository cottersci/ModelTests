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
import linear

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
parser.add_argument('--Nnoise',
                    type=int,
                    default=64,
                    help='''The number of noise dimenions to add to the
                            generator input''')
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
if(opt.Net == 'linear'):
    raise NotImplmentedError("Linear networks are not implmented")
    # G = linear.Net_G(opt.z_dim)
    # D = linear.Net_D()
else:
    S = conv.Net_Shared()
    G = conv.Net_G(opt.z_dim + opt.Nnoise)
    D = conv.Net_D(S)
    Q = conv.Net_Q(opt.z_dim,S,on_gpu=opt.no_cuda)

# Training Variables
opto_G = optim.Adam(G.parameters(), lr=opt.lr) # Training
opto_Q = optim.Adam(Q.parameters(), lr=opt.lr)
opto_D = optim.Adam(D.parameters(), lr=opt.lr) # Training

log = SummaryWriter(opt.logdir,opt.comment) # Logging
batch = batcher() # Logging

noise = torch.FloatTensor(opt.batch_size,opt.Nnoise)
gauss = torch.FloatTensor(opt.batch_size,opt.z_dim) # z ~ p(z)

##
### ---  FUNCTION DEFINITIONS -------------------------------- ####
##
def logQ(z,mu,logstd):
    std = logstd.exp()
    epislon = (z - mu).div(std + EPS)

    return -0.5 * math.log(2 * math.pi) - torch.log(std + EPS) - 0.5 * epislon.pow(2)

def train(epoch):
    '''
        Do 1 epoch of training, where an epoch is 1 iteration though the entire
        dataset.

        Args:
            epoch (int): number of epochs trained, used for logging
            noise (:class:`FloatTensor`): vector to hold the noise distribution
    '''
    D.train()
    G.train()
    for batch_idx in range(len(train_loader)):
        batch.batch()

        ####
        # Prepare batch
        img0, _ = next(train_loader.__iter__())
        if opt.no_cuda:
            img0 = img0.cuda()
        x_real  = Variable(img0)
        noise.resize_(x_real.size()[0],opt.Nnoise)
        gauss.resize_(img0.size()[0],opt.z_dim)

        ####
        # Train D
        opto_D.zero_grad()

        # P_D(x_real)
        p_real = D(x_real) + 1e-8
        D_real = -torch.mean( torch.log(p_real) )
        D_real.backward()

        # z ~ N(0,1), x_fake ~ q(x|z)
        gauss.normal_()
        noise.normal_()
        z = Variable(gauss)
        x_fake = G(torch.cat((z,Variable(noise)),1)) if opt.Nnoise > 0 else G(z)

        # P_D(x_fake)
        p_fake = D(x_fake) + 1e-8 #
        D_fake = -torch.mean( torch.log(1 - p_fake) )
        D_fake.backward()

        D_total = D_real + D_fake
        opto_D.step()

        batch.add('loss/total',D_total.item())
        batch.add('loss/real',D_real.item())
        batch.add('loss/fake',D_fake.item())

        ####
        # Train G and Q
        opto_G.zero_grad()
        opto_Q.zero_grad()

        # z ~ N(0,1), x_fake ~ q(x|z)
        gauss.normal_()
        noise.normal_()
        z = Variable(gauss,requires_grad=True)
        z.register_hook(lambda grad: batch.add('norms/G',grad.data.norm(2,1).mean()))
        x_fake = G(torch.cat((z,Variable(noise)),1)) if opt.Nnoise > 0 else G(z)

        #P_D(x_fake)
        x_fake.register_hook(lambda grad: batch.add('norms/D',grad.data.norm(2,1).mean()))
        p_fake = D(x_fake) + 1e-8
        G_loss = torch.mean( 1 - torch.log(p_fake) )
        G_loss.backward(retain_graph=True)

        #c ~ Q(x_fake)
        x_fake.register_hook(lambda grad: batch.add('norms/Q',grad.data.norm(2,1).mean()))
        c,mu,logstd = Q(x_fake)
        batch.add('debug/Qvar',logstd.exp().pow(2).mean())
        Q_loss = -logQ(z,mu,logstd).mean() # log(P(z|x))
        Q_loss.backward()

        opto_Q.step()
        opto_G.step()

        batch.add('loss/G',G_loss.item())
        batch.add('loss/auxiliary',Q_loss.item())

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
    noise.resize_(z.size()[0],opt.Nnoise).normal_()
    x = G.eval()(torch.cat((z,Variable(noise)),1)) if opt.Nnoise > 0 else G(z)

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
### ---  TRAINING ------------------------------------------------ ####
##
if(opt.no_cuda):
    G = G.cuda()
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
                         'G': G
                        })

folder_name = './CHECKPOINT-END'
util.save_nets(folder_name,
                {
                 'D': D,
                 'G': G
                })
