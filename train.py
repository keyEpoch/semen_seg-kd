# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from dataset import TrainDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata

# 11.06
# train one epoch
# one epoch contains many iters
# segmentation_modules contains teacher_segmentation_modules & student_segmentation_modules
def train(segmentation_modules, iterator, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    # set segmentation_module to training mode
    segmentation_module.train(not args.fix_bn)

    # main loop
    tic = time.time()
    for i in range(args.epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        # clear previous gradients
        segmentation_modules[0].zero_grad()
        segmentation_modules[1].zero_grad()

        acc, output, stu_seg = fetch_outputs(segmentation_modules, batch_data, args)

        acc_tea = acc['tea']; acc_stu = acc['stu'];
        out_tea = output['tea']; out_stu = output['stu']

        # forward pass
        # 3 res: 总的 kd_loss, teacher 网络的精度, student 网络的精度
        kd_loss = loss_fn_kd(out_stu, stu_seg, out_tea, args)
  
        kd_loss = kd_loss.mean()
        acc_tea = acc_tea.mean()
        acc_stu = acc_stu.mean()
        
        # Backward
        kd_loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(kd_loss.data.item())
        ave_acc_tea.update(acc_tea.data.item()*100)
        ave_acc_stu.update(acc_stu.data.item()*100)

        # calculate accuracy, and display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy_tea: {:4.2f}, Accuracy_stu: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.running_lr_encoder, args.running_lr_decoder,
                          ave_acc_tea.average(), ave_acc_stu.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(kd_loss.data.item())
            history['train']['acc_tea'].append(acc_tea.data.item())
            history['train']['acc_stu'].append(acc_stu.data.item())

        # adjust learning rate
        cur_iter = i + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, args)

def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    # dict_encoder_save = {k: v for k, v in dict_encoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}
    # dict_decoder_save = {k: v for k, v in dict_decoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}
    
    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_encoder,
               '{}/encoder_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder,
               '{}/decoder_{}'.format(args.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=args.lr_encoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr_encoder = args.lr_encoder * scale_running_lr
    args.running_lr_decoder = args.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = args.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = args.running_lr_decoder

# 11.05
# 在 main 函数里构造 tea_segmentation_module 和 stu_segmentation_module
# 合并成 segmentation_modules, 0是 teacher, 1是student
def fetch_outputs(segmentation_modules, batch_data, args):    # fetch a batch data
    acc = {}; output = {};
    _,tea_acc,tea_output,_ = tea_segmentation_module[0](batch_data)
    _,stu_acc,stu_output,stu_seg = stu_segmentation_module[1](batch_data)
    acc['tea'] = tea_acc; acc['stu'] = stu_acc
    output['tea'] = tea_output; output['stu'] = stu_output
    return acc, output, stu_seg

# 10.31
# compute the total loss
def loss_fn_kd(outputs, labels, teacher_outputs, args):
    # two hyper-parameters 
    alpha = args.alpha
    T = args.temperature

    soft_stu = F.log_softmax(outputs/T, dim = 1)
    soft_tea = F.softmax(teacher_outputs/T, dim = 1)

    # KL 散度
    # KL divergence is a useful distance measure for continuous distributions
    KD_loss = nn.KLDivLoss()(soft_stu, soft_tea)*(alpha*T*T) + F.cross_entropy(outputs, labels)*(1. - alpha)

    # return the final loss
    return KD_loss

def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder)

    crit = nn.NLLLoss(ignore_index=-1)
    # crit = loss_fn_kd()

    if args.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, args.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)

    loader_train = torchdata.DataLoader(
        dataset_train,
        batch_size=args.num_gpus,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True)

    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if args.num_gpus > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=range(args.num_gpus))
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(segmentation_module, iterator_train, optimizers, history, epoch, args)

        # checkpointing
        checkpoint(nets, history, args, epoch)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--weights_encoder', default='',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder', default='',
                        help="weights to finetune net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/train.odgt')
    parser.add_argument('--list_val',
                        default='./data/validation.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=8, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=5000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--deep_sup_scale', default=0.4, type=float,
                        help='the weight of deep supervision loss')
    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_class', default=19, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=[300,375,450,525,600], nargs='+', type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.max_iters = args.epoch_iters * args.num_epoch
    args.running_lr_encoder = args.lr_encoder
    args.running_lr_decoder = args.lr_decoder

    args.id += '-' + str(args.arch_encoder)
    args.id += '-' + str(args.arch_decoder)
    args.id += '-ngpus' + str(args.num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgMaxSize' + str(args.imgMaxSize)
    args.id += '-paddingConst' + str(args.padding_constant)
    args.id += '-segmDownsampleRate' + str(args.segm_downsampling_rate)
    args.id += '-LR_encoder' + str(args.lr_encoder)
    args.id += '-LR_decoder' + str(args.lr_decoder)
    args.id += '-epoch' + str(args.num_epoch)
    args.id += '-decay' + str(args.weight_decay)
    args.id += '-fixBN' + str(args.fix_bn)
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)









# 10.31
def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder)

    crit = nn.NLLLoss(ignore_index=-1)

    if args.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, args.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)

    loader_train = torchdata.DataLoader(
        dataset_train,
        batch_size=args.num_gpus,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True)

    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if args.num_gpus > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=range(args.num_gpus))
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(segmentation_module, iterator_train, optimizers, history, epoch, args)

        # checkpointing
        checkpoint(nets, history, args, epoch)

    print('Training Done!')