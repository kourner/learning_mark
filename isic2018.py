from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import custom_transforms as tr

class ISICSegmentation(Dataset):
    """
    ISIC2018 dataset
    """
    NUM_CLASSES = 2

    def __init__(self, args, split='train', ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._root = 'D:/long-term/Pycharm_project/unet/Image_Segmentation-master/dataset'
        self._image_dir = os.path.join(self._root, split)
        self._cat_dir = self._image_dir + '_GT/'
        self.split = [split]


        # if isinstance(split, str):
        #     self.split = [split]
        # else:
        #     split.sort()
        #     self.split = split

        self.args = args
        self.im_ids = []
        self.images = []
        self.categories = []

        self.image_paths = list(map(lambda x: os.path.join(self._image_dir, x), os.listdir(self._image_dir)))

        for _, image_path in enumerate(self.image_paths):
            _image = image_path

            filename = image_path.split('_')[-1][:-len(".jpg")]
            _cat = self._cat_dir + 'ISIC_' + filename + '_segmentation.png'
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.im_ids.append(image_path)
            self.images.append(_image)
            self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:

            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)
            # # elif split == 'test':
            #     return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
