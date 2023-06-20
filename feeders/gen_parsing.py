import os
import numpy as np
import PIL.Image as Image

def post_process(pred):
    h, w, _ = pred.shape
    pred = pred.reshape((h, w))
    img = Image.fromarray(pred)
    return img

def gen_featuremap(filename, random_interval = False, temporal_rgb_frames = 5):
    New_image = Image.new('RGB', size = (480, 480))
    pred_path = '../Human_Parsing/output/' + filename + '.npy'
    if not os.path.exists(pred_path):
        return New_image
    
    data = np.load(pred_path, allow_pickle = True)
    num_frames = data.shape[0]

    start = 0
    sample_interval = num_frames // temporal_rgb_frames
    if random_interval: sample_interval = np.random.randint(1, num_frames // temporal_rgb_frames + 1)
    if sample_interval == 0:
        Func = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        frame_range = Func(temporal_rgb_frames, num_frames)
    else: frame_range = range(start, num_frames, sample_interval)

    for idx, value in enumerate(frame_range[0:temporal_rgb_frames]):
        img = post_process(data[value]).resize((int(480/temporal_rgb_frames), 480))
        New_image.paste(img, (int(480/temporal_rgb_frames)*idx, 0))

    return New_image