import os
import numpy as np
import PIL.Image as Image

def post_process(pred):
    h, w, _ = pred.shape
    pred = pred.reshape((h, w))
    for i in range(256):
        for j in range(256):
            if pred[i][j] != 0:
                pred[i][j] += 50  # 20
    img = Image.fromarray(pred)
    return img

def gen_parsing(filename, random_interval = False, temporal_rgb_frames = 5):
    path = './output/' + filename + '.npy'
    if not os.path.exists(path):
        return False
    data = np.load(path, allow_pickle = True)

    # process each frame
    # for idx, value in enumerate(data):
    #     post_process(value)

    # view one frame
    img = post_process(data[0])
    img.show()

if __name__ == "__main__":
    filename = 'S001C001P001R001A001'
    stat = gen_parsing(filename, random_interval = True, temporal_rgb_frames=5)
    
    print("All done!")    
