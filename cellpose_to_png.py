import numpy as np
import cv2
import os

def main():
    data_dir = 'data'
    save_dir = 'images'

    npy_files = [os.path.join(data_dir, s) for s in os.listdir(data_dir) if s.endswith('.npy')]
    
    for file in npy_files:
        print(file)
        fname = file.split('.')[0]
        fname = fname.split('/')[-1]
        items = np.load(file, allow_pickle=True).item()
        mask = (items['masks'][:, :, None]  > 0) * 1.0
        outline = (items['outlines'][:, :, None]  > 0) * 1.0
        mask = mask * (1.0 - outline) * 255
        mask = mask[:,:,0]
        image = items['img']

        impath = os.path.join(save_dir, fname + "_image.png")
        maskpath = os.path.join(save_dir, fname + "_mask.png")
        print(impath)
        print(maskpath)
        cv2.imwrite(impath, image)
        cv2.imwrite(maskpath, mask)

if __name__ == '__main__':
    main()