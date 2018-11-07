# CUDA_VISIBLE_DEVICES=4,5 python3 eval.py --id baseline-resnet50_dilated8-ppm_bilinear_deepsup-ngpus2-batchSize4-imgMaxSize1000-paddingConst8-segmDownsampleRate8-LR_encoder0.02-LR_decoder0.02-epoch20-decay0.0001-fixBN0 #--suffix SUFFIX # --arch_encoder resnet50 --arch_decoder upernet --padding_constant 32 --num_class 150  # --visualize

CUDA_VISIBLE_DEVICES=4 python eval.py --id baseline-resnet18_dilated8-ppm_bilinear_deepsup-ngpus2-batchSize16-imgMaxSize1000-paddingConst8-segmDownsampleRate8-LR_encoder0.01-LR_decoder0.01-epoch20-decay0.0001-fixBN0 --arch_encoder resnet18_dilated8 --fc_dim 512 --num_class 20

