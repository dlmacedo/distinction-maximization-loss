from __future__ import print_function
import argparse
import torch
import models
import os
import losses
import loaders
import torch
import metrics
import numpy as np
import scipy.optimize as opt

parser = argparse.ArgumentParser(description='Calibrator')
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
    print("###############################")
    print("###############################")
    print("######### CALIBRATION #########")
    print("###############################")
    print("###############################")
    print(args)
    
    dir_path = os.path.join("experiments", args.dir, "train_classify", "data~"+args.dataset+"+model~"+args.net_type+"+loss~"+str(args.loss))
    file_path = os.path.join(dir_path, "results_calib.csv")

    with open(file_path, "w") as results_file:
        results_file.write("EXECUTION,MODEL,DATA,LOSS,ECE,TEMPERATURE\n")
   
    # define number of classes
    if args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 10

    image_loaders = loaders.ImageLoader(args)
    _, _, testloader = image_loaders.get_loaders()

    for args.execution in range(1, args.executions + 1):
        print("\nEXECUTION:", args.execution)
        pre_trained_net = os.path.join(dir_path, "model" + str(args.execution) + ".pth")

        if args.loss.split("_")[0] == "softmax":
            loss_first_part = losses.SoftMaxLossFirstPart
        elif args.loss.split("_")[0] == "isomax":
            loss_first_part = losses.IsoMaxLossFirstPart
        elif args.loss.split("_")[0] == "isomaxplus":
            loss_first_part = losses.IsoMaxPlusLossFirstPart
        elif args.loss.split("_")[0] == "dismax":
            loss_first_part = losses.DisMaxLossFirstPart

        # load networks
        if args.net_type == 'resnet34':
            net_trained = models.ResNet34(num_c=args.num_classes, loss_first_part=loss_first_part)
        elif args.net_type == 'densenetbc100':
            net_trained = models.DenseNet3(100, int(args.num_classes), loss_first_part=loss_first_part)
        elif args.net_type == "wideresnet2810":
            net_trained = models.Wide_ResNet(depth=28, widen_factor=10, num_classes=args.num_classes, loss_first_part=loss_first_part)
        net_trained.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        net_trained.eval()
        print('load net_trained: ' + args.net_type)
        
        ece_criterion = metrics.ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        model = net_trained.cuda()

        with torch.no_grad():
            for input, label in testloader:
                input = input.cuda()
                logits = model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        logits = logits.cpu()
        labels = labels.cpu()

        def ece_eval(tempearature):
            loss = ece_criterion.loss(logits.numpy()/tempearature,labels.numpy(),15)
            return loss

        ##########################################################################################################################
        temperature_for_min_ece, min_ece, _ = opt.fmin_l_bfgs_b(ece_eval, np.array([1.0]), approx_grad=True, bounds=[(0.001,100)])
        ##########################################################################################################################

        print("########################################")
        print("Min ECE", min_ece)
        print("Temperature for Min ECE", temperature_for_min_ece[0])
        print("########################################")

        with open(file_path, "a") as results_file:
            results_file.write("{},{},{},{},{},{}\n".format(
                str(args.execution), args.net_type, args.dataset, str(args.loss), min_ece, temperature_for_min_ece[0]))


if __name__ == '__main__':
    main()
