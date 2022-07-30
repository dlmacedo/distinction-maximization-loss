#!/usr/bin/env bash

python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~wideresnet2810+loss~softmax_std_std_std" -gpu 0 -x 10 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~wideresnet2810+loss~isomax_std_std_std" -gpu 0 -x 10 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~wideresnet2810+loss~isomaxplus_std_std_std" -gpu 0 -x 10 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~wideresnet2810+loss~dismax_std_std_std" -gpu 0 -x 10 -e 90 -lrde "30 60"
python main.py -ei="all" -et "train_classify" -ec "data~tinyimagenet+model~wideresnet2810+loss~dismax_std_std_fpr" -gpu 0 -x 10 -e 90 -lrde "30 60"

#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~wideresnet2810+loss~softmax_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~wideresnet2810+loss~isomax_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~wideresnet2810+loss~isomaxplus_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~wideresnet2810+loss~dismax_std_std_std" -gpu 0
#python main.py -ei="all" -et "extract_ood_logits_metrics" -ec "data~tinyimagenet+model~wideresnet2810+loss~dismax_std_std_fpr" -gpu 0

python calibrate.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss softmax_std_std_std --gpu 0 -x 10
python calibrate.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss isomax_std_std_std --gpu 0 -x 10
python calibrate.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss isomaxplus_std_std_std --gpu 0 -x 10
python calibrate.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss dismax_std_std_std --gpu 0 -x 10
python calibrate.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss dismax_std_std_fpr --gpu 0 -x 10

python detect.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss softmax_std_std_std --gpu 0 -x 10
python detect.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss isomax_std_std_std --gpu 0 -x 10
python detect.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss isomaxplus_std_std_std --gpu 0 -x 10
python detect.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss dismax_std_std_std --gpu 0 -x 10
python detect.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss dismax_std_std_fpr --gpu 0 -x 10

# if dismax was trained with precompute_thresholds=True, the code bellow may be used to verify them.
#python verify.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss dismax_std_std_std --gpu 0 -x 10
#python verify.py --dir all --dataset tinyimagenet --net_type wideresnet2810 --loss dismax_std_std_fpr --gpu 0 -x 10
