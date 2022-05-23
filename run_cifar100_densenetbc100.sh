#!/usr/bin/env bash

python main.py -ei="all" -et "train_classify" -ec "data~cifar100+model~densenetbc100+loss~softmax_std_std_std" -gpu 0 -x 10
python main.py -ei="all" -et "train_classify" -ec "data~cifar100+model~densenetbc100+loss~isomax_std_std_std" -gpu 0 -x 10
python main.py -ei="all" -et "train_classify" -ec "data~cifar100+model~densenetbc100+loss~isomaxplus_std_std_std" -gpu 0 -x 10
python main.py -ei="all" -et "train_classify" -ec "data~cifar100+model~densenetbc100+loss~dismax_std_std_std" -gpu 0 -x 10

python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~cifar100+model~densenetbc100+loss~softmax_std_std_std" -gpu 0
python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~cifar100+model~densenetbc100+loss~isomax_std_std_std" -gpu 0
python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~cifar100+model~densenetbc100+loss~isomaxplus_std_std_std" -gpu 0
python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~cifar100+model~densenetbc100+loss~dismax_std_std_std" -gpu 0

python calibrate.py --dir all --dataset cifar100 --net_type densenetbc100 --loss softmax_std_std_std --gpu 0 -x 10
python calibrate.py --dir all --dataset cifar100 --net_type densenetbc100 --loss isomax_std_std_std --gpu 0 -x 10
python calibrate.py --dir all --dataset cifar100 --net_type densenetbc100 --loss isomaxplus_std_std_std --gpu 0 -x 10
python calibrate.py --dir all --dataset cifar100 --net_type densenetbc100 --loss dismax_std_std_std --gpu 0 -x 10

python detect.py --dir all --dataset cifar100 --net_type densenetbc100 --loss softmax_std_std_std --gpu 0 -x 10
python detect.py --dir all --dataset cifar100 --net_type densenetbc100 --loss isomax_std_std_std --gpu 0 -x 10
python detect.py --dir all --dataset cifar100 --net_type densenetbc100 --loss isomaxplus_std_std_std --gpu 0 -x 10
python detect.py --dir all --dataset cifar100 --net_type densenetbc100 --loss dismax_std_std_std --gpu 0 -x 10

# if dismax was trained with precompute_thresholds=True, the code bellow may be used to verify them.
#python verify.py --dir all --dataset cifar100 --net_type densenetbc100 --loss dismax_std_std_std --gpu 0 -x 10
