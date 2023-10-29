# Author: Jan Niklas Kolf, 2020
from face_image_quality import SER_FIQ
import cv2
import argparse
import os, sys
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--dir_path', default='/vinai/quandm7/evaluation_dreambooth/Celeb/5/', help='path to input image file')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def main():
    # Sample code of calculating the score of an image
    
    # Create the SER-FIQ Model
    # Choose the GPU, default is 0.
    args = parse_args()
    ser_fiq = SER_FIQ(gpu=0)
    dir_files = os.listdir(args.dir_path)
    ave_score = 0
    score_list = {}
    for file in tqdm(dir_files):
        img = cv2.imread(os.path.join(args.dir_path, file))
        # Align the image
        aligned_img = ser_fiq.apply_mtcnn(img)
        # Calculate the quality score of the image
        # T=100 (default) is a good choice
        # Alpha and r parameters can be used to scale your
        # score distribution.
        score = ser_fiq.get_score(aligned_img, T=100)
        ave_score += score
        score_list[file] = score
        
    print(ave_score/len(dir_files))
    print(score_list)



if __name__ == '__main__':
    main()


