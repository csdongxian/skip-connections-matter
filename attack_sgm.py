import argparse
import torch
import torch.nn as nn
from torchvision import transforms


import pretrainedmodels
from advertorch.attacks import LinfPGDAttack, MomentumIterativeAttack
from utils_sgm import register_hook_for_resnet, register_hook_for_densenet
from utils_data import SubsetImageNet, save_images


model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack Evaluation')
parser.add_argument('--input_dir', default='./SubImageNet224', help='the path of original dataset')
parser.add_argument('--output_dir', default='./save', help='the path of the saved dataset')
parser.add_argument('--arch', default='resnet18',
                    help='source model for black-box attack evaluation',
                    choices=model_names)
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=16, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--print_freq', default=10, type=int)


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
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]


def generate_adversarial_example(model, data_loader, adversary, img_path):
    """
    evaluate model by black-box attack
    """
    model.eval()

    for batch_idx, (inputs, true_class, idx) in enumerate(data_loader):
        inputs, true_class = \
            inputs.to(device), true_class.to(device)

        # attack
        inputs_adv = adversary.perturb(inputs, true_class)

        save_images(inputs_adv.detach().cpu().numpy(), img_list=img_path,
                    idx=idx, output_dir=args.output_dir)
        # assert False
        if batch_idx % args.print_freq == 0:
            print('generating: [{0}/{1}]'.format(batch_idx, len(data_loader)))


def main():
    # create dataloader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_set = SubsetImageNet(root=args.input_dir, transform=transform_test)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    # create models
    net = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    model = nn.Sequential(Normalize(mean=net.mean, std=net.std), net)
    model = model.to(device)
    model.eval()

    # create adversary attack
    epsilon = args.epsilon / 255.0
    if args.step_size < 0:
        step_size = epsilon / args.num_steps
    else:
        step_size = args.step_size / 255.0

    # if args.gamma < 1.0:
    #     print('using our method')
    #     register_hook(model, args.arch, args.gamma, is_conv=args.is_conv)

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
                                  rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)

    generate_adversarial_example(model=model, data_loader=data_loader,
                                 adversary=adversary, img_path=data_set.img_path)


if __name__ == '__main__':
    main()
