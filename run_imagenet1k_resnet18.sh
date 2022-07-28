#!/usr/bin/env bash

python main.py -ei="all" -et "train_classify" -ec "data~imagenet1k+model~resnet18+loss~softmax_std_std_std" -gpu 0 -x 1 -bs 256 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~imagenet1k+model~resnet18+loss~isomax_std_std_std" -gpu 0 -x 1 -bs 256 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~imagenet1k+model~resnet18+loss~isomaxplus_std_std_std" -gpu 0 -x 1 -bs 256 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~imagenet1k+model~resnet18+loss~dismax_std_std_std" -gpu 0 -x 1 -bs 256 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~imagenet1k+model~resnet18+loss~dismax_std_std_fpr" -gpu 0 -x 1 -bs 256 -e 90 -lrde "30 60"

#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~imagenet1k+model~resnet18+loss~softmax_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~imagenet1k+model~resnet18+loss~isomax_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~imagenet1k+model~resnet18+loss~isomaxplus_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~imagenet1k+model~resnet18+loss~dismax_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~imagenet1k+model~resnet18+loss~dismax_std_std_fpr" -gpu 0

python calibrate.py --dir all --dataset imagenet1k --net_type resnet18 --loss softmax_std_std_std --gpu 0 -x 1 -bs 256
python calibrate.py --dir all --dataset imagenet1k --net_type resnet18 --loss isomax_std_std_std --gpu 0 -x 1 -bs 256
python calibrate.py --dir all --dataset imagenet1k --net_type resnet18 --loss isomaxplus_std_std_std --gpu 0 -x 1 -bs 256
python calibrate.py --dir all --dataset imagenet1k --net_type resnet18 --loss dismax_std_std_std --gpu 0 -x 1 -bs 256
python calibrate.py --dir all --dataset imagenet1k --net_type resnet18 --loss dismax_std_std_fpr --gpu 0 -x 1 -bs 256

python detect.py --dir all --dataset imagenet1k --net_type resnet18 --loss softmax_std_std_std --gpu 0 -x 1 -bs 256
python detect.py --dir all --dataset imagenet1k --net_type resnet18 --loss isomax_std_std_std --gpu 0 -x 1 -bs 256
python detect.py --dir all --dataset imagenet1k --net_type resnet18 --loss isomaxplus_std_std_std --gpu 0 -x 1 -bs 256
python detect.py --dir all --dataset imagenet1k --net_type resnet18 --loss dismax_std_std_std --gpu 0 -x 1 -bs 256
python detect.py --dir all --dataset imagenet1k --net_type resnet18 --loss dismax_std_std_fpr --gpu 0 -x 1 -bs 256

## if dismax was trained with precompute_thresholds=True, the code bellow may be used to verify them.
##python verify.py --dir all --dataset imagenet1k --net_type resnet18 --loss dismax_std_std_std --gpu 0 -x 1
##python verify.py --dir all --dataset imagenet1k --net_type resnet18 --loss dismax_std_std_fpr --gpu 0 -x 1
