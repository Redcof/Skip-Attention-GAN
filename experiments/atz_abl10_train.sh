# experiment with KK, CK, and CL classes with F1, F2, M1, M2 subjects
# 128x128 patch database. customdataset/atz/atz_patch_dataset__3_128_36.csv
# batch size 32

# start visdom in a separate terminal
#visdom

# =-=-=-=-=-=-=-=-=-=-
export PYTHONPATH="$PYTHONPATH:$(pwd)/preprocessing"
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONPATH="$PYTHONPATH:$(pwd)/experiment"
## ablation study
#python preprocessing/split_atz.py --ablation 10

# full data experiment
# python preprocessing/split_atz.py
export DATA_ROOT="/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages" # mac
#export DATA_ROOT="C:\\Users\\dndlssardar\\Downloads\\THZ_dataset_det_VOC\\JPEGImages"                            # windows
export DATA_ROOT="/mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages"                             # wsl
# run training
python train.py \
  --phase train \
  \
  --name abl_train_128x128_20230114 \
  --model skipattentionganomaly \
  \
  --dataroot "$DATA_ROOT" \
  --dataset atz \
  --atz_patch_db customdataset/atz/atz_patch_dataset__3_128_36_v2_10%_30_99%.csv \
  --area_threshold 0.05 \
  --atz_subjects "['F1', 'M1', 'F2', 'M2']" \
  --atz_classes '["KK", "CK", "CL", "MD", "SS", "GA"]' \
  --manualseed 47 \
  \
  --verbose \
  --display\
  \
  --device gpu \
  --gpu_ids "[0,]" \
  \
  --isize 128 \
  --batchsize 8 \
  --verbose \
  \
  --niter 20 \
  --iter 0 \
  \
  --save_image_freq 5 \
  --print_freq 5 \
  --atz_ablation 24

echo "Training done."
date
