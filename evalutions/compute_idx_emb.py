import numpy as np
import os
from deepface import DeepFace
import argparse


# def compute_idx_embedding(path):
#     """
#     Compute the embedding of each person given images
#     """
#     ave_embedding = 0
#     count_file = 0
#     for file in os.listdir(path):
#         if file.endswith(".jpg") or file.endswith(".png"):
#             count_file += 1
#             embedding_objs = DeepFace.represent(img_path = os.path.join(path, file), model_name="ArcFace", detector_backend="retinaface", align=True)
#             embedding = embedding_objs[0]["embedding"]
#             embedding = np.array(embedding)
#             ave_embedding += embedding
#     ave_embedding /= count_file
#     np.save(os.path.join(path, "embedding.npy"), ave_embedding)
#     return 0

def compute_idx_embedding(paths):
    """
    Compute the embedding of each person given images
    """
    print(paths)
    ave_embedding = 0
    count_file = 0
    for path in paths:
        for file in os.listdir(path):
            if file.endswith(".jpg") or file.endswith(".png"):
                try:
                   count_file += 1
                   embedding_objs = DeepFace.represent(img_path = os.path.join(path, file), model_name="ArcFace", detector_backend="retinaface", align=True)
                   embedding = embedding_objs[0]["embedding"]
                   embedding = np.array(embedding)
                   ave_embedding += embedding
                except:
                    print(">>>>>>> SKIP", file)
    ave_embedding /= count_file
    return ave_embedding


def parse_args():
    parser = argparse.ArgumentParser(description='compute embedding for each person')
    parser.add_argument('--img_dir', type=str, default='', required=True, help='path to datadir')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    compute_idx_embedding(args.img_dir)
    
if __name__ == '__main__':
    main()