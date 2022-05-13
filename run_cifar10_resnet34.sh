#!/usr/bin/env bash

python main.py -ei="all" -et "train_classify" -ec "data~cifar10+model~resnet34+loss~softmax_std_std_std" -gpu 0 -x 5
python main.py -ei="all" -et "train_classify" -ec "data~cifar10+model~resnet34+loss~isomax_std_std_std" -gpu 0 -x 5
python main.py -ei="all" -et "train_classify" -ec "data~cifar10+model~resnet34+loss~isomaxplus_std_std_std" -gpu 0 -x 5
python main.py -ei="all" -et "train_classify" -ec "data~cifar10+model~resnet34+loss~dismax_std_std_std" -gpu 0 -x 5

python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~cifar10+model~resnet34+loss~softmax_std_std_std" -gpu 0
python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~cifar10+model~resnet34+loss~isomax_std_std_std" -gpu 0
python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~cifar10+model~resnet34+loss~isomaxplus_std_std_std" -gpu 0
python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~cifar10+model~resnet34+loss~dismax_std_std_std" -gpu 0

python calibrate.py --dir all --dataset cifar10 --net_type resnet34 --loss softmax_std_std_std --gpu 0 -x 5
python calibrate.py --dir all --dataset cifar10 --net_type resnet34 --loss isomax_std_std_std --gpu 0 -x 5
python calibrate.py --dir all --dataset cifar10 --net_type resnet34 --loss isomaxplus_std_std_std --gpu 0 -x 5
python calibrate.py --dir all --dataset cifar10 --net_type resnet34 --loss dismax_std_std_std --gpu 0 -x 5

python detect.py --dir all --dataset cifar10 --net_type resnet34 --loss softmax_std_std_std --gpu 0 -x 5
python detect.py --dir all --dataset cifar10 --net_type resnet34 --loss isomax_std_std_std --gpu 0 -x 5
python detect.py --dir all --dataset cifar10 --net_type resnet34 --loss isomaxplus_std_std_std --gpu 0 -x 5
python detect.py --dir all --dataset cifar10 --net_type resnet34 --loss dismax_std_std_std --gpu 0 -x 5

# if dismax was trained with precompute_thresholds=True, the code bellow may be used to verify them.
#python verify.py --dir all --dataset cifar10 --net_type resnet34 --loss dismax_std_std_std --gpu 0 -x 5
