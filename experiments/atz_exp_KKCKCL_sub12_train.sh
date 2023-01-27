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
export DATA_ROOT="/mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages" # wsl
# run training
python train.py \
--area_threshold 0.05 \
--atz_ablation 0 \
--atz_classes "[]" \
--atz_patch_db customdataset/atz/atz_patch_dataset__3_128_36_v2_10%_30_99%.csv \
--atz_patch_overlap 0.2 \
--atz_subjects "[]" \
--atz_wavelet "{'wavelet':'sym4', 'method':'VisuShrink','level':3, 'mode':'hard'}" \
--atz_wavelet_denoise \
--batchsize 128 \
--dataroot /mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages \
--dataset atz \
--device cuda \
--display \
--isize 128 \
--lr 0.0002 \
--manualseed 47 \
--metric roc \
--model skipattentionganomaly \
--name exp14_contd13_128x128_20230114 \
--niter 50 \
--iter 7 \
--load_weights output/exp13_128x128_20230114/train/weights/auc618\
--outf ./output \
--phase train \
--print_freq 1280 \
--save_image_freq 1280 \
--save_test_images \
--verbose \
#--atz_train_txt customdataset/atz/train.txt \
#--atz_test_txt  customdataset/atz/trest.txt \
echo "Training done."
date
