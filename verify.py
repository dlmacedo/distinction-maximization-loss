from __future__ import print_function
import argparse
import torch
import models
import os
import losses
import data_loader
from torchvision import transforms
import numpy
import models
import torchmetrics

numpy.set_printoptions(edgeitems=5, linewidth=160, formatter={'float': '{:0.6f}'.format})
torch.set_printoptions(edgeitems=5, precision=6, linewidth=160)

parser = argparse.ArgumentParser(description='OOD Performance Estimation Verifier')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
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
    print("#########################")
    print("#########################")
    print("######## VERIFY #########")
    print("#########################")
    print("#########################")
    print(args)

    dir_path = os.path.join("experiments", args.dir, "train_classify", "data~"+args.dataset+"+model~"+args.net_type+"+loss~"+str(args.loss))

    file_path = os.path.join(dir_path, "results_threshold.csv")
    with open(file_path, "w") as results_file:
        results_file.write("EXECUTION,MODEL,IN-DATA,OUT-DATA,LOSS,SCORE,PERCENTILE,FPR,TPR,THRESHOLD\n")

    file_path2 = os.path.join(dir_path, "results_auroc.csv")
    with open(file_path2, "w") as results_file:
        results_file.write("EXECUTION,MODEL,IN-DATA,OUT-DATA,LOSS,SCORE,AUROC\n")
  
    # define number of classes
    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
    else:
        args.num_classes = 10

    if args.dataset == 'cifar10':
        out_dist_list = ['cifar100', 'imagenet_resize', 'lsun_resize', 'svhn']
    elif args.dataset == 'cifar100':
        out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize', 'svhn']

    if args.dataset == 'cifar10':
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])
    elif args.dataset == 'cifar100':
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))])

    for args.execution in range(1, args.executions + 1):    
        print("\nEXECUTION:", args.execution)
        pre_trained_net = os.path.join(dir_path, "model" + str(args.execution) + ".pth")

        if args.loss.split("_")[0] == "softmax":
            loss_first_part = losses.SoftMaxLossFirstPart
            scores = ["ES"]
        elif args.loss.split("_")[0] == "isomax":
            loss_first_part = losses.IsoMaxLossFirstPart
            scores = ["ES"]
        elif args.loss.split("_")[0] == "isomaxplus":
            loss_first_part = losses.IsoMaxPlusLossFirstPart
            scores = ["MDS"]
        elif args.loss.split("_")[0] == "dismax":
            loss_first_part = losses.DisMaxLossFirstPart
            scores = ["MMLES"]

        # load networks
        if args.net_type == 'resnet34':
            model = models.ResNet34(num_c=args.num_classes, loss_first_part=loss_first_part)
        elif args.net_type == 'densenetbc100':
            model = models.DenseNet3(100, int(args.num_classes), loss_first_part=loss_first_part)
        elif args.net_type == "wideresnet2810":
            model = models.Wide_ResNet(depth=28, widen_factor=10, num_classes=args.num_classes, loss_first_part=loss_first_part)
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        model.cuda()
        print('load model: ' + args.net_type)

        model.eval()
        print(model.classifier.validationset_available.item())

        # load dataset
        print('load target valid data: ', args.dataset)
        _, test_loader = data_loader.getTargetDataSet(args, args.dataset, args.batch_size, in_transform, args.dataroot)

        for score in scores:
            print("###############################")
            print("###############################")
            print("SCORE:", score)
            print("###############################")
            print("###############################")

            i = 1 if model.classifier.validationset_available.item() else 0
            print("Precomputed Thresholds (initial):\n", model.classifier.precomputed_thresholds[i].detach().cpu().numpy())

            for out_dist in out_dist_list:
                out_test_loader = data_loader.getNonTargetDataSet(args, out_dist, args.batch_size, in_transform, args.dataroot)
                print('Out-distribution: ' + out_dist)
                auroc, roc = detect(model, test_loader, out_test_loader, score)
                print("AUROC: ", auroc.item())
                fpr_list = []
                tpr_list = []
                threshold_list = []
                
                for j in range(25):
                    index = find_index_of_nearest(model.classifier.precomputed_thresholds[i,j].detach().cpu().numpy(), roc[2].detach().cpu().numpy())
                    fpr_list.append(roc[0][index].item())
                    tpr_list.append(roc[1][index].item())
                    threshold_list.append(roc[2][index].item())
                print("FPRs:\n", fpr_list)
                print("TPRs:\n", tpr_list)
                print("Precomputed Threshold (final):\n", threshold_list)

                #Saving regular estimations (requires in-distribution data and out-distribution data):
                with open(file_path, "a") as results_file:
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "0", fpr_list[0], tpr_list[0], 0))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "0.1", fpr_list[1], tpr_list[1], 1))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "0.2", fpr_list[2], tpr_list[2], 2))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "0.3", fpr_list[3], tpr_list[3], 3))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "0.4", fpr_list[4], tpr_list[4], 4))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "0.5", fpr_list[5], tpr_list[5], 5))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "1", fpr_list[6], tpr_list[6], 6))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "2", fpr_list[7], tpr_list[7], 7))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "3", fpr_list[8], tpr_list[8], 8))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "4", fpr_list[9], tpr_list[9], 9))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "5", fpr_list[10], tpr_list[10], 10))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "10", fpr_list[11], tpr_list[11], 11))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "15", fpr_list[12], tpr_list[12], 12))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "20", fpr_list[13], tpr_list[13], 13))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "25", fpr_list[14], tpr_list[14], 14))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "30", fpr_list[15], tpr_list[15], 15))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "35", fpr_list[16], tpr_list[16], 16))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "40", fpr_list[17], tpr_list[17], 17))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "45", fpr_list[18], tpr_list[18], 18))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "50", fpr_list[19], tpr_list[19], 19))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "60", fpr_list[20], tpr_list[20], 20))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "70", fpr_list[21], tpr_list[21], 21))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "80", fpr_list[22], tpr_list[22], 22))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "90", fpr_list[23], tpr_list[23], 23))
                    results_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, "100", fpr_list[24], tpr_list[24], 24))

                with open(file_path2, "a") as results_file:
                    results_file.write("{},{},{},{},{},{},{}\n".format(str(args.execution), args.net_type, args.dataset, out_dist, str(args.loss), score, auroc))


def detect(model, inloader, oodloader, score_type):
    auroc = torchmetrics.AUROC(pos_label=1)
    roc = torchmetrics.ROC(pos_label=1)
    model.eval()

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(inloader):
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            targets.fill_(1)
            logits = model(inputs)

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

            auroc.update(scores, targets)
            roc.update(scores, targets)

        for _, (inputs, targets) in enumerate(oodloader):
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            targets.fill_(0)
            logits = model(inputs)

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

            auroc.update(scores, targets)            
            roc.update(scores, targets)
        
    return auroc.compute(), roc.compute()


def find_index_of_nearest(value, array):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return idx


if __name__ == '__main__':
    main()
