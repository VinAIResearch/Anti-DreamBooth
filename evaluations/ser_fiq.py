import cv2
import argparse
import os
from FaceImageQuality.face_image_quality import SER_FIQ

def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--prompt_path', default='/vinai/quandm7/evaluation_dreambooth/Celeb/5/', help='path to input image file')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

prompts = ["a_dslr_portrait_of_sks_person", "a_photo_of_a_sks_person"]

def main():
    args = parse_args()
    ser_fiq = SER_FIQ(gpu=args.gpu)
    prompt_path = args.prompt_path
    prompt_score = 0
    count = 0
    
    for img_name in os.listdir(prompt_path):
        if "png" in img_name or "jpg" in img_name:
            img_path = os.path.join(prompt_path, img_name)
            img = cv2.imread(img_path)
            aligned_img = ser_fiq.apply_mtcnn(img)
            if aligned_img is not None:
                score = ser_fiq.get_score(aligned_img, T=100)
                prompt_score+=score
                count += 1
                                     
    return prompt_score/count



if __name__ == '__main__':
    fia_score = main()
    print("FIQ score: {}".format(fia_score))