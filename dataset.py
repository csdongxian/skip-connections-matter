import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from scipy.misc import imread, imsave


class CustomDataSet(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.image_list = os.listdir(input_dir)

    def __getitem__(self, item):
        img_path = self.image_list[item]
        with open(os.path.join(self.input_dir, img_path), 'rb') as f:
            image = imread(f, mode='RGB').transpose((2, 0, 1)).astype(np.float32) / 255.0
        return image, item

    def __len__(self):
        return len(self.image_list)


def load_images(input_dir, batch_size):
    """Read png images from input directory in batches.
        Args:
            input_dir: input directory
            batch_size: shape of minibatch array, i.e. [batch_size, height, width, 3]
        Return:
            dataloader
    """
    img_set = CustomDataSet(input_dir=input_dir)
    img_loader = DataLoader(img_set, batch_size=batch_size, num_workers=2)
    return img_loader, img_set.image_list


def save_images(images, img_list, idx, output_dir):
    """Saves images to the output directory.
        Args:
          images: tensor with minibatch of images
          img_list: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
          output_dir: directory where to save images
    """
    for i, sample_idx in enumerate(idx.numpy()):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        filename = img_list[sample_idx]
        cur_images = (images[i, :, :, :] * 255).astype(np.uint8)
        with open(os.path.join(output_dir, filename), 'wb') as f:
            imsave(f, cur_images.transpose(1, 2, 0), format='png')


if __name__ == '__main__':
    cdataset = CustomDataSet('nat_images')
    cdataset.__getitem__(0)
