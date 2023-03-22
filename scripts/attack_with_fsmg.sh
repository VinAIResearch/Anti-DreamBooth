export EXPERIMENT_NAME="FSMG"
export MODEL_PATH="./stable-diffusion/stable-diffusion-2-1-base"
export CLASS_DIR="data/class-person"
export CLEAN_TRAIN_DIR="data/n000050/set_A"
export REF_MODEL_PATH="outputs/$EXPERIMENT_NAME/n000050_REFERENCE"


# ------------------------- Train DreamBooth model on set A -------------------------
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$CLEAN_TRAIN_DIR\
  --class_data_dir=$CLASS_DIR \
  --output_dir=$REF_MODEL_PATH \
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


# ------------------------- Train FSMG on set B -------------------------
export CLEAN_ADV_DIR="data/n000050/set_B"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_ADVERSARIAL"

mkdir -p $OUTPUT_DIR
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

accelerate launch attacks/fsmg.py \
  --pretrained_model_name_or_path="$REF_MODEL_PATH/checkpoint-1000"  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$CLEAN_ADV_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100 \
  --checkpointing_steps=20 \
  --center_crop \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2


# ------------------------- Train DreamBooth on perturbed examples -------------------------
export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/100"
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