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

python histogram_view.py \
  --hist_csv "$1" \
  --phase train --model skipattentionganomaly --dataroot $DATA_ROOT --dataset atz \
  --atz_patch_db customdataset/atz/atz_patch_dataset__3_128_36_v2_10%_30_99%.csv \
  --area_threshold 0.05 \
  --atz_subjects "['F1', 'M1', 'F2', 'M2']" \
  --atz_classes "['KK', 'CK', 'CL', 'MD', 'SS', 'GA']" \
  --manualseed 47 \
  --isize 128 --batchsize 128
