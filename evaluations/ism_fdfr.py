from deepface import DeepFace
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
from compute_idx_emb import compute_idx_embedding


def compute_face_embedding(img_path):
    """Extract face embedding vector of given image
    Args:
        img_path (str): path to image
    Returns:
        None: no face found
        vector: return the embedding of biggest face among the all found faces
    """
    try:
        resps = DeepFace.represent(img_path = os.path.join(img_path), 
                                   model_name="ArcFace", 
                                   enforce_detection=True, 
                                   detector_backend="retinaface", 
                                   align=True)
        if resps == 1:
            # detect only 1 face
            return np.array(resps[0]["embedding"])
        else:
            # detect more than 1 faces, choose the biggest one
            resps = list(resps)
            resps.sort(key=lambda resp: resp["facial_area"]["h"]*resp["facial_area"]["w"], reverse=True)
            return np.array(resps[0]["embedding"])
    except Exception:
        # no face found
        return None

def get_precomputed_embedding(path):
    """Get face embedding by loading the path to numpy file
    Args:
        path (str): path to numpy file 
    Returns:
        vector: face embedding
    """
    return np.load(path)


# def matching_score_id(image_path, id_emb_path):
#     """getting the matching score between face image and precomputed embedding
#     Args:
#         img (2D images): images
#         emb (vector): face embedding
#     Returns:
#         None: cannot detect face from img
#         int: identity score matching
#     """
#     image_emb = compute_face_embedding(image_path)
#     id_emb = get_precomputed_embedding(id_emb_path)
#     if image_emb is None:
#         return None
#     image_emb, id_emb = torch.Tensor(image_emb), torch.Tensor(id_emb)
#     ism = F.cosine_similarity(image_emb, id_emb, dim=0)
#     return ism

def matching_score_id(image_path, avg_embedding):
    """getting the matching score between face image and precomputed embedding

    Args:
        img (2D images): images
        emb (vector): face embedding

    Returns:
        None: cannot detect face from img
        int: identity score matching
    """
    image_emb = compute_face_embedding(image_path)
    id_emb = avg_embedding
    if image_emb is None:
        return None
    image_emb, id_emb = torch.Tensor(image_emb), torch.Tensor(id_emb)
    ism = F.cosine_similarity(image_emb, id_emb, dim=0)
    return ism

# def matching_score_genimage_id(images_path, id_emb_path):
#     image_list = os.listdir(images_path)
#     fail_detection_count = 0
#     ave_ism = 0
#     for image_name in image_list:
#         image_path = os.path.join(images_path, image_name)
#         ism = matching_score_id(image_path, id_emb_path)
#         if ism is None:
#             fail_detection_count += 1
#         else:
#             ave_ism += ism
#     return ave_ism/(len(image_list)-fail_detection_count), fail_detection_count/len(image_list)

def matching_score_genimage_id(images_path, list_id_path):
    image_list = os.listdir(images_path)
    fail_detection_count = 0
    ave_ism = 0
    avg_embedding = compute_idx_embedding(list_id_path)

    for image_name in image_list:
        image_path = os.path.join(images_path, image_name)
        ism = matching_score_id(image_path, avg_embedding)
        if ism is None:
            fail_detection_count += 1
        else:
            ave_ism += ism
    if fail_detection_count != len(image_list):
        return ave_ism/(len(image_list)-fail_detection_count), fail_detection_count/len(image_list)
    return None, 1

def parse_args():
    parser = argparse.ArgumentParser(description='FDFR and ISM evaluation')
    parser.add_argument('--data_dir', type=str, default='', required=True, help='path to datadir')
    parser.add_argument('--emb_dirs', metavar='N', type=str, nargs='+', help='list of paths to clean image')
    # parser.add_argument('--emb_dir', type=str, default='', required=True, help='path to embedding dir')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ism, fdr = matching_score_genimage_id(args.data_dir, args.emb_dirs)
    print("ISM and FDR are {} and {}".format(ism, fdr))
    
if __name__ == '__main__':
    main()

