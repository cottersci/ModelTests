import torch
from pytorch_utils import init
import math

def parse_args():
    parser = init.parser()
    parser.add_argument('z_dim', type=int, help='Z dimension')
    parser.add_argument('--download',action='store_true',help='Download Dataset')
    parser.add_argument('--dataset',
                        default="MNIST",
                        help='choices: MNIST, SVHN. default=MNIST')
    parser.add_argument('data_loc',
                        default=".",
                        help='Location of/to download dataset. default=./')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5,
                        help='Learning rate for D and G networks, default=1e-5')
    parser.add_argument('--Nimages',
                        type=float,
                        default=2,
                        help='''The test phase will produce a grid
                                    of Nimages^z_dim images''')
    parser.add_argument('--noise-dim',
                        type=int,
                        default=64,
                        help='''The number of noise dimenions to add to the
                                generator input''')
    return parser.parse_args()

def init_dataset_loader(dataset,batch_size,location, **kwargs):
    """
        Initilize the dataset loader

        Args:
            dataset (str): the dataset to load. Options:
                MNIST: the mnist dataset
                SVHN: Grayscale SVHN dataset, resized to the same size as MNIST
            batch_size (int): batch size
            location (str): Path to dir to store/load datset data
            kwargs: all other keyword args are passed to to the loader initilizer
    """
    if(dataset == 'MNIST'):
        return init.mnist(batch_size,
                                  location,
                                  train=True,
                                  download=True,
                                  **kwargs)
    elif(dataset == "SVHN"):
        return init.svhn(batch_size,
                                 location,
                                 train=True,
                                 download=True,
                                 **kwargs)
    else:
        print('Invalid --dataset choice')
        exit()

def logQ(z, mu, logstd, EPS = 1e-8):
    std = logstd.exp()
    epislon = (z - mu).pow(2).div(2 * std + EPS)

    return -0.5 * (math.log(2 * math.pi) + torch.log(std + EPS) + epislon)

def grad_norm(parameters,norm_type = 2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm

def clip_grad_norm(parameters, g_u, g_m):
    """
        Clip the paramaters

        g_m norm of paranters to clip
        g_u norm to clipe paramters to
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        p.grad.data.div_(g_m).mul_(min(g_u,g_m))
