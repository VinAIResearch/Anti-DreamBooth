export EXPERIMENT_NAME="E-FSMG"
export MODEL_PATH="./stable-diffusion/stable-diffusion-2-1-base"
export CLASS_DIR="data/class-person"
export CLEAN_TRAIN_DIR="data/n000050/set_A"

# ------------------------- Train DreamBooth models on set A -------------------------
# stable diffusion
sd14_path="./stable-diffusion/stable-diffusion-v1-4"
sd15_path="./stable-diffusion/stable-diffusion-v1-5"
sd21_path="./stable-diffusion/stable-diffusion-2-1-base"
sd_paths=($sd14_path $sd15_path $sd21_path)

# ref models
ref_sd14_root="dreambooth-clean-outputs/V14_VGG512_set_A/n000050_REFERENCE/"
ref_sd15_root="dreambooth-clean-outputs/V15_VGG512_set_A/n000050_REFERENCE/"
ref_sd21_root="dreambooth-clean-outputs/V21_VGG512_set_A/n000050_REFERENCE/"
ref_paths=($ref_sd14_root $ref_sd15_root $ref_sd21_root)

versions=(V14 V15 V21)

for ((i=0;i<3;i++));
do
    echo ${ref_paths[$i]}
    echo ${sd_paths[$i]}
    echo ${versions[$i]}

    accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path=${sd_paths[$i]}  \
        --enable_xformers_memory_efficient_attention \
        --train_text_encoder \
        --instance_data_dir=$CLEAN_TRAIN_DIR\
        --class_data_dir="${CLASS_DIR}_${versions[$i]}" \
        --output_dir=${ref_paths[$i]} \
        --with_prior_preservation \
        --prior_loss_weight=1.0 \
        --instance_prompt="a photo of sks person" \
        --class_prompt="a photo of person" \
        --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
        --resolution=512 \
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=500 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=16
done


# ------------------------- Train E-FSMG on set B -------------------------
export CLEAN_ADV_DIR="data/n000050/set_B"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_ADVERSARIAL"

mkdir -p $OUTPUT_DIR
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

# pretrained sd models on clean set A
ref_sd14_path="${ref_sd14_root}/checkpoint-1000"
ref_sd15_path="${ref_sd15_root}/checkpoint-1000"
ref_sd21_path="${ref_sd21_root}/checkpoint-1000"
ref_model_paths="${ref_sd14_path},${ref_sd15_path},${ref_sd21_path}"

accelerate launch attacks/ensemble_fsmg.py \
  --pretrained_model_name_or_path=${ref_model_paths} \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$CLEAN_ADV_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=60 \
  --max_adv_train_steps=12 \
  --checkpointing_steps=20 \
  --center_crop \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2


# ------------------------- Train DreamBooth on perturbed examples -------------------------
export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/60"
export DREAMBOOTH_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_DREAMBOOTH"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --checkpointing_steps=500 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=16