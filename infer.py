import argparse
import os

import torch
from diffusers import StableDiffusionPipeline


parser = argparse.ArgumentParser(description="Inference")
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions are saved",
)

args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)

    # define prompts
    prompts = [
        "a photo of a sks person",
        "a dslr portrait of sks person",
        "a close-up photo of sks person riding a bike",
        "a photo of sks person in front of eiffel tower",
        "a selfie photo of sks person on top of mount fuji",
    ]

    # create & load model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
    ).to("cuda")

    for prompt in prompts:
        print(">>>>>>", prompt)
        norm_prompt = prompt.lower().replace(",", "").replace(" ", "_")
        out_path = f"{args.output_dir}/{norm_prompt}"
        os.makedirs(out_path, exist_ok=True)
        for i in range(2):
            images = pipe([prompt] * 8, num_inference_steps=100, guidance_scale=7.5).images
            for idx, image in enumerate(images):
                image.save(f"{out_path}/{i}_{idx}.png")
    del pipe
    torch.cuda.empty_cache()
