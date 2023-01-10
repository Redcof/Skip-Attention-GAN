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
export DATA_ROOT="/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages"
# run training
python train.py \
  --phase train \
  \
  --name abstudy_7_128x128_20230111 \
  --model skipattentionganomaly \
  \
  --dataroot "$DATA_ROOT" \
  --atz_patch_db customdataset/atz/atz_patch_dataset__3_128_36.csv\
  --atz_train_txt customdataset/atz/set_ablation_10/atz_dataset_train_ablation_10.txt\
  --atz_test_txt customdataset/atz/set_ablation_10/atz_dataset_test_ablation_10.txt\
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
  --niter 10 \
  \
  --save_image_freq 1 \
  --print_freq 1

echo "Training done."
echo "Testing start."
date

python train.py \
  --phase test \
  \
  --name abstudy_7_128x128_20230111 \
  --model skipattentionganomaly \
  \
  --dataroot "$DATA_ROOT" \
  --atz_patch_db customdataset/atz/atz_patch_dataset__3_128_36.csv\
  --atz_train_txt customdataset/atz/set_ablation_10/atz_dataset_train_ablation_10.txt\
  --atz_test_txt customdataset/atz/set_ablation_10/atz_dataset_test_ablation_10.txt\
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
  --niter 10 \
  \
  --save_image_freq 1 \
  --print_freq 1

date