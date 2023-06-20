import torch
import pickle
import numpy as np
from PIL import ImageFile
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
from feeders import gen_parsing

class Feeder(torch.utils.data.Dataset):
    def __init__(self, sample_path, label_path, random_interval=False, temporal_rgb_frames=5, debug=False):
        self.debug = debug
        self.sample_path = sample_path
        self.label_path = label_path
        self.random_interval = random_interval
        self.temporal_rgb_frames = temporal_rgb_frames

        self.load_data()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        self.sample_name = np.loadtxt(self.sample_path, dtype = str)
        self.label = np.loadtxt(self.label_path, dtype = int)
        if self.debug:
            self.label = self.label[0:100]
            self.sample_name = self.sample_name[0:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        label = self.label[index]
        filename = self.sample_name[index]
        img = gen_parsing.gen_featuremap(filename, self.random_interval, self.temporal_rgb_frames)
        width, height = img.size
        img = np.array(img.getdata())
        img = torch.from_numpy(img).float()
        _, C = img.size()
        img = img.permute(1, 0).contiguous()
        img = img.view(C, height, width)
        img = self.transform(img)
        return img, label
