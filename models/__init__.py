# Contains code based on https://github.com/tamarott/SinGAN
import torch


from .generator import GrowingGenerator
from .discriminator import Discriminator
from .layout_discriminator import LayoutDiscriminator2D


def weights_init(m):
    """Initialize convolution and normalization layers with GAN defaults."""
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Conv3d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Norm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_G(opt):
    """Create a generator and optionally restore its checkpoint."""

    G = GrowingGenerator(opt).to(opt.device)
    G.apply(weights_init)
    if opt.netG != "":
        G.load_state_dict(torch.load(opt.netG, map_location=opt.device))
    return G


def init_D(opt):
    """Create a representation discriminator and optionally restore it."""

    D = Discriminator(opt).to(opt.device)
    D.apply(weights_init)
    if opt.netD != "":
        D.load_state_dict(torch.load(opt.netD, map_location=opt.device))
    print(D)
    return D


def init_D_layout(opt):
    """Create the 2D layout discriminator configured for the current run."""

    D = LayoutDiscriminator2D(
        in_channels=opt.layout_channels,
        nfc=opt.layout_nfc,
    ).to(opt.device)
    D.apply(weights_init)
    return D

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    MSGGan = False
    if  MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)#.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
