# start visdom in a separate terminal
#visdom

# =-=-=-=-=-=-=-=-=-=-
export PYTHONPATH="$PYTHONPATH:$(pwd)/preprocessing"
# ablation study
python preprocessing/split_atz.py --ablation 5

# full data experiment
# python preprocessing/split_atz.py

# run training
python train.py \
  --phase train \
  \
  --name abstudy_5_128x128_20230108 \
  --model skipattentionganomaly \
  \
  --dataroot "/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages" \
  --manualseed 47 \
  --dataset atz \
  \
  --verbose \
  --display \
  \
  --device cpu \
  --gpu_ids "" \
  \
  --isize 128 \
  --batchsize 32 \
  --verbose \
  \
  --niter 2 \
  \
  --save_image_freq 1 \
  --print_freq 1
