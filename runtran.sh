# CUDA_VISIBLE_DEVICES=4,5 python train.py --num_gpus 2 --arch_encoder resnet18_dilated8 --batch_size_per_gpu 4 --epoch_iters 10000 --lr_encoder 1e-2 --lr_decoder 1e-2 --fc_dim 512
#CUDA_VISIBLE_DEVICES=2,3,4,5 python train.py --num_gpus 2 --arch_encoder resnet34_dilated8 --batch_size_per_gpu 2 --epoch_iters 10000 --lr_encoder 1e-2 --lr_decoder 1e-2
CUDA_VISIBLE_DEVICES=4,6 python train.py  --num_gpus 2 --arch_encoder resnet18_dilated8 --batch_size_per_gpu 8 --epoch_iters 1000 --lr_encoder 1e-2 --lr_decoder 1e-2 --fc_dim 512 --num_class 20
