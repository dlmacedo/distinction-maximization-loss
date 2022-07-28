from __future__ import print_function
import argparse
import torch
import models
import os
import losses
import data_loader
import calculate_log as callog
from torchvision import transforms
import timm
from tqdm import tqdm

parser = argparse.ArgumentParser(description='OOD Detector')
parser.add_argument('-bs', '--batch-size', type=int, default=64, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | tinyimagenet')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--net_type', required=True, help='resnet | wideresnet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--loss', required=True, help='the loss used')
parser.add_argument('--dir', default="", type=str, help='Part of the dir to use')
parser.add_argument('-x', '--executions', default=1, type=int, metavar='N', help='Number of executions (default: 1)')

args = parser.parse_args()
torch.cuda.set_device(args.gpu)


def main():
    print("\n\n\n\n\n")
    print("#############################")
    print("#############################")
    print("######### DETECTION #########")
    print("#############################")
    print("#############################")
    print(args)

    dir_path = os.path.join("experiments", args.dir, "train_classify", "data~"+args.dataset+"+model~"+args.net_type+"+loss~"+str(args.loss))
    file_path = os.path.join(dir_path, "results_odd.csv")

    with open(file_path, "w") as results_file:
        results_file.write(
            "EXECUTION,MODEL,IN-DATA,OUT-DATA,LOSS,AD-HOC,SCORE,INFER-LEARN,INFER-TRANS,"
            "TNR,AUROC,DTACC,AUIN,AUOUT,CPU_FALSE,CPU_TRUE,GPU_FALSE,GPU_TRUE,TEMPERATURE,MAGNITUDE\n")

    args_outf = os.path.join("temp", "ood", args.loss, args.net_type + '+' + args.dataset)
    if os.path.isdir(args_outf) == False:
        os.makedirs(args_outf)
    
    # define number of classes
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.data_type = "image"
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.data_type = "image"
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        args.data_type = "image"
    elif args.dataset == 'imagenet1k':
        args.num_classes = 1000
        args.data_type = "image"

    if args.dataset == 'cifar10':
        out_dist_list = ['cifar100', 'imagenet_resize', 'lsun_resize', 'svhn']
    elif args.dataset == 'cifar100':
        out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize', 'svhn']
    elif args.dataset == 'tinyimagenet':
        out_dist_list = ['imagenet-o-64', 'cifar10_64', 'cifar100_64', 'svhn_64']
    elif args.dataset == 'imagenet1k':
        out_dist_list = ['imagenet-o']

    if args.dataset == 'cifar10':
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])
    elif args.dataset == 'cifar100':
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))])
    elif args.dataset == 'tinyimagenet':
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    elif args.dataset == 'imagenet1k':
        args.dataroot = '/mnt/ssd/imagenet1k'
        args.input_size = 224
        args.DEFAULT_CROP_RATIO = 0.875
        in_transform = transforms.Compose([
            transforms.Resize(int(args.input_size / args.DEFAULT_CROP_RATIO)),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    for args.execution in range(1, args.executions + 1):    
        print("\nEXECUTION:", args.execution)
        pre_trained_net = os.path.join(dir_path, "model" + str(args.execution) + ".pth")

        if args.loss.split("_")[0] == "softmax":
            loss_first_part = losses.SoftMaxLossFirstPart
            scores = ["MPS"]
        elif args.loss.split("_")[0] == "isomax":
            loss_first_part = losses.IsoMaxLossFirstPart
            scores = ["ES"]
        elif args.loss.split("_")[0] == "isomaxplus":
            loss_first_part = losses.IsoMaxPlusLossFirstPart
            scores = ["MDS"]
        elif args.loss.split("_")[0] == "dismax":
            loss_first_part = losses.DisMaxLossFirstPart
            scores = ["MMLES","MPS"]

        # load networks
        if args.net_type == 'resnet34':
            model = models.ResNet34(num_c=args.num_classes, loss_first_part=loss_first_part)
        elif args.net_type == 'densenetbc100':
            model = models.DenseNet3(100, int(args.num_classes), loss_first_part=loss_first_part)
        elif args.net_type == "wideresnet2810":
            model = models.Wide_ResNet(depth=28, widen_factor=10, num_classes=args.num_classes, loss_first_part=loss_first_part)
        elif args.net_type == "resnet18":
            model = timm.create_model('resnet18', pretrained=False)
            num_in_features = model.get_classifier().in_features
            model.fc = loss_first_part(num_in_features, args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        model.cuda()
        print('load model: ' + args.net_type)
        
        # load dataset
        print('load target valid data: ', args.dataset)
        _, test_loader = data_loader.getTargetDataSet(args, args.dataset, args.batch_size, in_transform, args.dataroot)

        for score in scores:
            print("###############################")
            print("###############################")
            print("SCORE:", score)
            print("###############################")
            print("###############################")
            base_line_list = []
            print("In-distribution")
            get_scores(args, model, test_loader, args_outf, True, score)
            out_count = 0
            for out_dist in out_dist_list:
                print('Out-distribution: ' + out_dist)
                out_test_loader = data_loader.getNonTargetDataSet(args, out_dist, args.batch_size, in_transform, args.dataroot)
                get_scores(args, model, out_test_loader, args_outf, False, score)
                test_results = callog.metric(args_outf, ['PoT'])
                base_line_list.append(test_results)
                out_count += 1
            
            # print the results
            mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
            print('Baseline method: train in_distribution: ' + args.dataset + '==========')
            count_out = 0
            for results in base_line_list:
                print('out_distribution: '+ out_dist_list[count_out])
                for mtype in mtypes:
                    print(' {mtype:6s}'.format(mtype=mtype), end='')
                print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
                print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
                print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
                print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
                print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
                with open(file_path, "a") as results_file:
                    results_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist_list[count_out],
                        str(args.loss), "NATIVE", score, 'NO', False,
                        '{:.2f}'.format(100.*results['PoT']['TNR']),
                        '{:.2f}'.format(100.*results['PoT']['AUROC']),
                        '{:.2f}'.format(100.*results['PoT']['DTACC']),
                        '{:.2f}'.format(100.*results['PoT']['AUIN']),
                        '{:.2f}'.format(100.*results['PoT']['AUOUT']),
                        0, 0, 0, 0, 1, 0))
                count_out += 1


def get_scores(args, model, test_loader, outf, out_flag, score_type=None):
    print("===>>> get scores <<<===")
    model.eval()
    total = 0
    if out_flag == True:
        temp_file_name_val = '%s/confidence_PoV_In.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_In.txt'%(outf)
    else:
        temp_file_name_val = '%s/confidence_PoV_Out.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_Out.txt'%(outf)     
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')

    for batch_index, batch_data in enumerate(tqdm(test_loader)):
        data = batch_data[0]
        batch_size = data.size(0)

        total += batch_size
        data = data.cuda()  
        with torch.no_grad():
            logits = model(data)
            probabilities = torch.nn.Softmax(dim=1)(logits)
            if score_type == "MPS":
                scores = probabilities.max(dim=1)[0]
            elif score_type == "ES":
                scores = (probabilities * torch.log(probabilities)).sum(dim=1)
            elif score_type == "MDS":
                scores = logits.max(dim=1)[0]
            elif score_type == "MMLS":
                scores = logits.max(dim=1)[0] + logits.mean(dim=1)
            elif score_type == "MMLES":
                scores = logits.max(dim=1)[0] + logits.mean(dim=1) + (probabilities * torch.log(probabilities)).sum(dim=1)
            elif score_type == "MMLEPS":
                scores = logits.max(dim=1)[0] + logits.mean(dim=1) + (probabilities * torch.log(probabilities)).sum(dim=1) + probabilities.max(dim=1)[0]
        for i in range(batch_size):
            f.write("{}\n".format(scores[i]))

    f.close()
    g.close()


if __name__ == '__main__':
    main()
