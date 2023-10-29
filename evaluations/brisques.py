import argparse
import os
from PIL import Image
from brisque import BRISQUE

def parse_args():
    parser = argparse.ArgumentParser(description='Brisque')
    parser.add_argument('--prompt_path', default=None, help='path to input image file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    prompt_path = args.prompt_path
    obj = BRISQUE(url=False)
    prompt_score = 0
    count = 0
    for img_name in os.listdir(prompt_path):
        if "png" in img_name or "jpg" in img_name:
            img_path = os.path.join(prompt_path, img_name)
            img = Image.open(img_path)
            brisque_score = obj.score(img)
            print(brisque_score)
            prompt_score += brisque_score
            count += 1
    return prompt_score/count


if __name__ == '__main__':
    brisque = main()
    print("The brisque score is {}".format(brisque))