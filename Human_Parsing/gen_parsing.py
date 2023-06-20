import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from net.pspnet import PSPNet
from torchvision import transforms

Two_Body = ['050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060',
                '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116',
                '117', '118', '119', '120']

models = {
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
}

parser = argparse.ArgumentParser(description="Human parsing of EPP-Net")
parser.add_argument('--ntu60_path', type=str, default='./dataset/ntu60/')
parser.add_argument('--ntu120_path', type=str, default='./dataset/ntu120/')
parser.add_argument('--models-path', type=str, default='./checkpoints')
parser.add_argument('--backend', type=str, default='resnet101')
parser.add_argument('--num-classes', type=int, default=20)
parser.add_argument('--samples_txt_path', type=str, default='./ntu120.txt')
parser.add_argument('--output_path', type=str, default='./output/')
args = parser.parse_args()

def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        if not epoch == 'last':
            epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch

def get_transform():
    transform_image_list = [
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform_image_list)

def human_parsing(all_samples, output_root_path):
    # load model
    snapshot = os.path.join(args.models_path, args.backend, 'PSPNet_last') # model path
    net, _ = build_network(snapshot, args.backend)
    net.eval()
    # load data
    data_transform = get_transform()

    for _, name in enumerate(all_samples):
        print("processing " + name)
        label = name[-3:]
        if int(label) > 60: data_path = args.ntu120_path + name + '/'
        else: data_path = args.ntu60_path + name + '/'
        if label not in Two_Body:
            num_frames = len(os.listdir(data_path))
            M = 1
        else:
            num_frames = int(len(os.listdir(data_path)) / 3)
            M = 2
        
        sample_pred = np.zeros((num_frames, 256, 256, 1), dtype=np.uint8)
        for idx in range(num_frames):
            if M == 1:
                img_path = data_path + str(idx) + '.jpg'
            else:
                img_path = data_path + str(idx) + '_mp.jpg'
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path)
            img = data_transform(img).to(device = 0)
        
            # inference
            with torch.no_grad():
                pred = net(img.unsqueeze(dim=0))
                pred = pred.squeeze(dim=0).permute(1, 2, 0)
                _, pred = torch.max(pred, dim = 2)
                pred = np.asarray(pred.reshape(256, 256, 1).cpu().numpy(), dtype = np.uint8)

                sample_pred[idx] = pred

        np.save(output_root_path + name + '.npy', sample_pred)

if __name__ == '__main__':
    samples_txt_path = args.samples_txt_path
    all_samples = np.loadtxt(samples_txt_path, dtype=str) 
    output_root_path = args.output_path
    human_parsing(all_samples, output_root_path)
    