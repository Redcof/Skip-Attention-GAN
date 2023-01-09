# script running guide
# sh train.sh abstudy_6_128x128_20230108 "/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages"

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
  --name $1 \
  --model skipattentionganomaly \
  \
  --dataroot $2 \
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
