# experiment with KK, CK, and CL classes with F1, F2, M1, M2 subjects
# Images with object: 672. customdataset/atz/set_KK_CK_CL_sub_1_2/atz_dataset_train_KK_CK_CL_subject12.files.txt
# Images with NO object: 157. customdataset/atz/set_KK_CK_CL_sub_1_2/atz_dataset_test_KK_CK_CL_subject12.files.txt
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
export DATA_ROOT="/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages"
date
# run training
python train.py \
  --phase train \
  \
  --name exp1_128x128_20230111 \
  --model skipattentionganomaly \
  \
  --dataroot "$DATA_ROOT" \
  --dataset atz \
  --atz_patch_db customdataset/atz/atz_patch_dataset__3_128_36.csv\
  --atz_train_txt customdataset/atz/set_KK_CK_CL_sub_1_2/atz_dataset_train_KK_CK_CL_subject12.files.txt\
  --atz_test_txt customdataset/atz/set_KK_CK_CL_sub_1_2/atz_dataset_test_KK_CK_CL_subject12.files.txt\
  --manualseed 47 \
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
  --niter 20 \
  --iter 0 \
  \
  --save_image_freq 10 \
  --print_freq 10

echo "Training done."
echo "Testing start."
date
# run training
python train.py \
  --phase test \
  \
  --name exp1_128x128_20230111 \
  --model skipattentionganomaly \
  \
  --dataroot "$DATA_ROOT" \
  --dataset atz \
  --atz_patch_db customdataset/atz/atz_patch_dataset__3_128_36.csv\
  --atz_train_txt customdataset/atz/set_KK_CK_CL_sub_1_2/atz_dataset_train_KK_CK_CL_subject12.files.txt\
  --atz_test_txt customdataset/atz/set_KK_CK_CL_sub_1_2/atz_dataset_test_KK_CK_CL_subject12.files.txt\
  --manualseed 47 \
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
  --niter 20 \
  --iter 0 \
  \
  --save_image_freq 10 \
  --print_freq 10

date