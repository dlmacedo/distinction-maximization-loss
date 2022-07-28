#!/usr/bin/env bash

python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~resnet34+loss~softmax_std_std_std" -gpu 0 -x 5 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~resnet34+loss~isomax_std_std_std" -gpu 0 -x 5 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~resnet34+loss~isomaxplus_std_std_std" -gpu 0 -x 5 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~resnet34+loss~dismax_std_std_std" -gpu 0 -x 5 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~resnet34+loss~dismax_std_std_fpr" -gpu 0 -x 5 -e 90 -lrde "30 60"

#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~resnet34+loss~softmax_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~resnet34+loss~isomax_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~resnet34+loss~isomaxplus_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~resnet34+loss~dismax_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~resnet34+loss~dismax_std_std_fpr" -gpu 0

python calibrate.py --dir all --dataset tinyimagenet --net_type resnet34 --loss softmax_std_std_std --gpu 0 -x 5
python calibrate.py --dir all --dataset tinyimagenet --net_type resnet34 --loss isomax_std_std_std --gpu 0 -x 5
python calibrate.py --dir all --dataset tinyimagenet --net_type resnet34 --loss isomaxplus_std_std_std --gpu 0 -x 5
python calibrate.py --dir all --dataset tinyimagenet --net_type resnet34 --loss dismax_std_std_std --gpu 0 -x 5
python calibrate.py --dir all --dataset tinyimagenet --net_type resnet34 --loss dismax_std_std_fpr --gpu 0 -x 5

python detect.py --dir all --dataset tinyimagenet --net_type resnet34 --loss softmax_std_std_std --gpu 0 -x 5
python detect.py --dir all --dataset tinyimagenet --net_type resnet34 --loss isomax_std_std_std --gpu 0 -x 5
python detect.py --dir all --dataset tinyimagenet --net_type resnet34 --loss isomaxplus_std_std_std --gpu 0 -x 5
python detect.py --dir all --dataset tinyimagenet --net_type resnet34 --loss dismax_std_std_std --gpu 0 -x 5
python detect.py --dir all --dataset tinyimagenet --net_type resnet34 --loss dismax_std_std_fpr --gpu 0 -x 5

# if dismax was trained with precompute_thresholds=True, the code bellow may be used to verify them.
#python verify.py --dir all --dataset tinyimagenet --net_type resnet34 --loss dismax_std_std_std --gpu 0 -x 5
#python verify.py --dir all --dataset tinyimagenet --net_type resnet34 --loss dismax_std_std_fpr --gpu 0 -x 5
