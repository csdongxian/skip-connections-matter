import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import pretrainedmodels

from advertorch.attacks import LinfPGDAttack, MomentumIterativeAttack

from dataset import *
from utils_sgm import register_hook_for_resnet, register_hook_for_densenet

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack')
parser.add_argument('--input-dir', default='', help='Input directory with images.')
parser.add_argument('--output-dir', default='', help='Output directory with images.')
parser.add_argument('--arch', default='densenet201', help='source model',
                    choices=model_names)
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=16, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--gamma', default=1.0, type=float)
parser.add_argument('--momentum', default=0.0, type=float)

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


def generate_adversarial_example(model, data_loader, adversary, img_path):
    """
    generate and save adversarial example
    """
    model.eval()

    for batch_idx, (inputs, idx) in enumerate(data_loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            _, pred = model(inputs).topk(1, 1, True, True)

        # craft adversarial images
        inputs_adv = adversary.perturb(inputs, pred.detach().view(-1))

        # save adversarial images
        save_images(inputs_adv.detach().cpu().numpy(), img_list=img_path,
                    idx=idx, output_dir=args.output_dir)


def main():
    # create model
    net = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    height, width = net.input_size[1], net.input_size[2]
    model = nn.Sequential(Normalize(mean=net.mean, std=net.std), net)
    model = model.to(device)

    # create dataloader
    data_loader, image_list = load_images(input_dir=args.input_dir, batch_size=args.batch_size,
                                          input_height=height, input_width=width)

    # create adversary
    epsilon = args.epsilon / 255.0
    if args.step_size < 0:
        step_size = epsilon / args.num_steps
    else:
        step_size = args.step_size / 255.0

    # using our method - Skip Gradient Method (SGM)
    if args.gamma < 1.0:
        if args.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            register_hook_for_resnet(model, arch=args.arch, gamma=args.gamma)
        elif args.arch in ['densenet121', 'densenet169', 'densenet201']:
            register_hook_for_densenet(model, arch=args.arch, gamma=args.gamma)
        else:
            raise ValueError('Current code only supports resnet/densenet. '
                             'You can extend this code to other architectures.')

    if args.momentum > 0.0:
        print('using PGD attack with momentum = {}'.format(args.momentum))
        adversary = MomentumIterativeAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                            eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                                            decay_factor=args.momentum,
                                            clip_min=0.0, clip_max=1.0, targeted=False)
    else:
        print('using linf PGD attack')
        adversary = LinfPGDAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                  eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                                  clip_min=0.0, clip_max=1.0, targeted=False)

    generate_adversarial_example(model=model, data_loader=data_loader,
                                 adversary=adversary, img_path=image_list)


if __name__ == '__main__':
    main()
