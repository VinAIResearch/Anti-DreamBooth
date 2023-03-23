import argparse
import copy
import hashlib
import itertools
import logging
import os
import warnings
from pathlib import Path

import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


logger = get_logger(__name__)


class DreamBoothDatasetFromTensor(Dataset):
    """Just like DreamBoothDataset, but take instance_images_tensor instead of path"""

    def __init__(
        self,
        instance_images_tensor,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_tensor = instance_images_tensor
        self.num_instance_images = len(self.instance_images_tensor)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_tensor[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_model(args, model_path):
    print(model_path)
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(model_path, args.revision)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        model_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=args.revision)

    # num_iters = 100
    # num_train_steps = 20
    # num_pgd_attack_steps = 20

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=args.revision)

    vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        print("You selected to used efficient xformers")
        print("Make sure to install the following packages before continue")
        print("pip install triton==2.0.0.dev20221031")
        print("pip install pip install xformers==0.0.17.dev461")

        unet.enable_xformers_memory_efficient_attention()

    return text_encoder, unet, tokenizer, noise_scheduler, vae


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir_for_train",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_data_dir_for_adversarial",
        type=str,
        default=None,
        required=True,
        help="A folder containing the images to add adversarial noise",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_f_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train surogate model.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=1.0 / 255,
        help="The step size for pgd.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=0.05,
        help="The noise budget for pgd.",
    )
    parser.add_argument(
        "--checkpointing_iterations",
        type=int,
        default=5,
        help=("Save a checkpoint of the training state every X iterations."),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def load_data(data_dir, size=512, center_crop=True) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    images = [image_transforms(Image.open(i).convert("RGB")) for i in list(Path(data_dir).iterdir())]
    images = torch.stack(images)
    return images


def train_one_epoch(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    num_steps=20,
):
    # Load the tokenizer

    unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])
    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    train_dataset = DreamBoothDatasetFromTensor(
        data_tensor,
        args.instance_prompt,
        tokenizer,
        args.class_data_dir,
        args.class_prompt,
        args.resolution,
        args.center_crop,
    )

    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    for step in range(num_steps):
        unet.train()
        text_encoder.train()

        step_data = train_dataset[step % len(train_dataset)]
        pixel_values = torch.stack([step_data["instance_images"], step_data["class_images"]]).to(
            device, dtype=weight_dtype
        )
        input_ids = torch.cat([step_data["instance_prompt_ids"], step_data["class_prompt_ids"]], dim=0).to(device)

        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # with prior preservation loss
        if args.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = instance_loss + args.prior_loss_weight * prior_loss

        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad()
        print(
            f"Step #{step}, loss: {loss.detach().item()}, prior_loss: {prior_loss.detach().item()}, instance_loss: {instance_loss.detach().item()}"
        )

    return [unet, text_encoder]


def pgd_attack(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    original_images: torch.Tensor,
    num_steps: int,
):
    """Return new perturbed data"""

    unet, text_encoder = models
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    perturbed_images = data_tensor.detach().clone()
    perturbed_images.requires_grad_(True)

    input_ids = tokenizer(
        args.instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.repeat(len(data_tensor), 1)

    for step in range(num_steps):
        perturbed_images.requires_grad = True
        latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor  # N=4, C, 64, 64

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids.to(device))[0]

        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # no prior preservation
        unet.zero_grad()
        text_encoder.zero_grad()
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss.backward()

        alpha = args.pgd_alpha
        eps = args.pgd_eps

        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=+eps)
        perturbed_images = torch.clamp(original_images + eta, min=-1, max=+1).detach_()
        print(f"PGD loss - step {step}, loss: {loss.detach().item()}")
    return perturbed_images


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.mixed_precision == "fp32":
                torch_dtype = torch.float32
            elif args.mixed_precision == "fp16":
                torch_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    clean_data = load_data(
        args.instance_data_dir_for_train,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    perturbed_data = load_data(
        args.instance_data_dir_for_adversarial,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    original_data = perturbed_data.clone()
    original_data.requires_grad_(False)

    model_paths = list(args.pretrained_model_name_or_path.split(","))
    num_models = len(model_paths)

    # MODEL_NAMES = ["text_encoder", "unet", "tokenizer", "noise_scheduler", "vae"]
    MODEL_BANKS = [load_model(args, path) for path in model_paths]
    MODEL_STATEDICTS = [
        {
            "text_encoder": MODEL_BANKS[i][0].state_dict(),
            "unet": MODEL_BANKS[i][1].state_dict(),
        }
        for i in range(num_models)
    ]

    for i in range(args.max_train_steps):
        en_data = 0.0
        for j, model_path in enumerate(model_paths):
            text_encoder, unet, tokenizer, noise_scheduler, vae = MODEL_BANKS[j]
            unet.load_state_dict(MODEL_STATEDICTS[j]["unet"])
            text_encoder.load_state_dict(MODEL_STATEDICTS[j]["text_encoder"])
            f = [unet, text_encoder]

            # 1. f' = f.clone()
            f_sur = copy.deepcopy(f)
            f_sur = train_one_epoch(
                args,
                f_sur,
                tokenizer,
                noise_scheduler,
                vae,
                clean_data,
                args.max_f_train_steps,
            )
            perturbed_data_f = pgd_attack(
                args,
                f_sur,
                tokenizer,
                noise_scheduler,
                vae,
                perturbed_data,
                original_data,
                args.max_adv_train_steps,
            )

            en_data += perturbed_data_f / num_models

            f = train_one_epoch(
                args,
                f,
                tokenizer,
                noise_scheduler,
                vae,
                perturbed_data_f,
                args.max_f_train_steps,
            )

            # save new statedicts
            MODEL_STATEDICTS[j]["unet"] = f[0].state_dict()
            MODEL_STATEDICTS[j]["text_encoder"] = f[1].state_dict()

            del f
            del text_encoder, unet, tokenizer, noise_scheduler, vae

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # update
        perturbed_data = en_data

        if (i + 1) % args.checkpointing_iterations == 0:
            save_folder = f"{args.output_dir}/noise-ckpt/{i+1}"
            os.makedirs(save_folder, exist_ok=True)
            noised_imgs = perturbed_data.detach()
            img_names = [
                str(instance_path).split("/")[-1]
                for instance_path in list(Path(args.instance_data_dir_for_adversarial).iterdir())
            ]
            for img_pixel, img_name in zip(noised_imgs, img_names):
                save_path = os.path.join(save_folder, f"{i+1}_noise_{img_name}")
                Image.fromarray(
                    (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ).save(save_path)
            print(f"Saved noise at step {i+1} to {save_folder}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
