export PYTHONPATH="$PYTHONPATH:$(pwd)/preprocessing"
# ablation study
python preprocessing/split_atz.py --ablation

# full data experiment
# python preprocessing/split_atz.py

# run training
python train.py                \
--phase train                  \
\
--name abstudy_3_20230105      \
--model skipattentionganomaly  \
\
--dataroot /Users/soumen/Desktop/Skip-Attention-GAN/customdataset/atz   \
--manualseed 47                \
--dataset atz                  \
\
--verbose                      \
--display                      \
\
--device cpu                   \
--gpu_ids ""                   \
\
--isize 512                    \
--batchsize 1                  \
--verbose                      \
\
--niter 5                      \
\
--save_test_images             \
--save_image_freq 1            \
--print_freq 1                 \

