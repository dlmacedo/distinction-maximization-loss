import os
import sys
import torch
import models
import loaders
import losses
import statistics
import math
import torchnet as tnt
import numpy as np
import utils
from torchvision import transforms
import timm

class ClassifierAgent:
    def __init__(self, args):
        self.args = args
        self.epoch = None

        image_loaders = loaders.ImageLoader(args)
        self.trainset_loader_for_train, self.trainset_loader_for_infer, self.valset_loader = image_loaders.get_loaders()
        print("\nDATASET:", args.dataset)

        if self.args.loss.split("_")[0] == "softmax":
            loss_first_part = losses.SoftMaxLossFirstPart
            loss_second_part = losses.SoftMaxLossSecondPart
        elif self.args.loss.split("_")[0] == "isomax":
            loss_first_part = losses.IsoMaxLossFirstPart
            loss_second_part = losses.IsoMaxLossSecondPart
        elif self.args.loss.split("_")[0] == "isomaxplus":
            loss_first_part = losses.IsoMaxPlusLossFirstPart
            loss_second_part = losses.IsoMaxPlusLossSecondPart
        elif self.args.loss.split("_")[0] == "dismax":
            loss_first_part = losses.DisMaxLossFirstPart
            loss_second_part = losses.DisMaxLossSecondPart
        else:
            sys.exit('You should pass a valid loss to use!!!')
        print("=> creating model '{}'".format(self.args.model_name))

        if self.args.model_name == "resnet34":
            self.model = models.ResNet34(num_c=self.args.number_of_model_classes, loss_first_part=loss_first_part)
            self.model_classifier = self.model.classifier
        elif self.args.model_name == "densenetbc100":
            self.model = models.DenseNet3(100, int(self.args.number_of_model_classes), loss_first_part=loss_first_part)
            self.model_classifier = self.model.classifier
        elif self.args.model_name == "wideresnet2810":
            self.model = models.Wide_ResNet(depth=28, widen_factor=10, num_classes=self.args.number_of_model_classes, loss_first_part=loss_first_part)
            self.model_classifier = self.model.classifier
        elif self.args.model_name == "resnet50":
            self.model = timm.create_model('resnet50', pretrained=False)
            print(self.model.default_cfg)
            num_in_features = self.model.get_classifier().in_features
            self.model.fc = loss_first_part(num_in_features, self.args.number_of_model_classes)
            self.model_classifier = self.model.fc
        elif self.args.model_name == "resnet18":
            self.model = timm.create_model('resnet18', pretrained=False)
            print(self.model.default_cfg)
            num_in_features = self.model.get_classifier().in_features
            self.model.fc = loss_first_part(num_in_features, self.args.number_of_model_classes)
            self.model_classifier = self.model.fc
        self.model.cuda()

        print("\nMODEL:", self.model)
        with open(os.path.join(self.args.experiment_path, 'model.arch'), 'w') as file:
            print(self.model, file=file)
        print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        utils.print_num_params(self.model)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
       
        self.criterion = loss_second_part(self.model_classifier)   
        if self.args.loss.split("_")[0].startswith("dismax"):
            if self.args.loss.split("_")[3].startswith("fpr"):
                self.criterion = loss_second_part(self.model_classifier, regularization="fpr")
            if self.args.loss.split("_")[3].startswith("0.5"):
                self.criterion = loss_second_part(self.model_classifier, regularization="fpr", alpha=0.5)
            if self.args.loss.split("_")[3].startswith("0.25"):
                self.criterion = loss_second_part(self.model_classifier, regularization="fpr", alpha=0.25)

        parameters = self.model.parameters()
        self.optimizer = torch.optim.SGD(parameters, lr=self.args.original_learning_rate, momentum=self.args.momentum, nesterov=True, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.learning_rate_decay_epochs, gamma=args.learning_rate_decay_rate)
        print("\nTRAIN:", self.criterion, self.optimizer, self.scheduler)

    def train_classify(self):
        if self.args.execution == 1:
            with open(self.args.executions_best_results_file_path, "w") as best_results:
                best_results.write(
                    "DATA,MODEL,LOSS,EXECUTION,EPOCH,"
                    "TRAIN LOSS,TRAIN ACC1,TRAIN SCALE,"
                    "TRAIN INTRA_LOGITS MEAN,TRAIN INTRA_LOGITS STD,TRAIN INTER_LOGITS MEAN,TRAIN INTER_LOGITS STD,"
                    "TRAIN MAX_PROBS MEAN,TRAIN MAX_PROBS STD,TRAIN ENTROPIES MEAN,TRAIN ENTROPIES STD,"
                    "VALID LOSS,VALID ACC1,VALID SCALE,"
                    "VALID INTRA_LOGITS MEAN,VALID INTRA_LOGITS STD,VALID INTER_LOGITS MEAN,VALID INTER_LOGITS STD,"
                    "VALID MAX_PROBS MEAN,VALID MAX_PROBS STD,VALID ENTROPIES MEAN,VALID ENTROPIES STD\n")
            with open(self.args.executions_raw_results_file_path, "w") as raw_results:
                raw_results.write("DATA,MODEL,LOSS,EXECUTION,EPOCH,SET,METRIC,VALUE\n")

        print("\n################ TRAINING AND VALIDATING ################")        
        best_model_results = {"VALID ACC1": 0}

        for self.epoch in range(1, self.args.epochs + 1):
            print("\n######## EPOCH:", self.epoch, "OF", self.args.epochs, "########")

            for param_group in self.optimizer.param_groups:
                print("\nLEARNING RATE:\t\t", param_group["lr"])                

            train_loss, train_acc1, train_scale, train_epoch_logits, train_epoch_metrics = self.train_epoch()           
            valid_loss, valid_acc1, valid_scale, valid_epoch_logits, valid_epoch_metrics = self.validate_epoch()
            
            if self.scheduler is not None:
                self.scheduler.step()

            train_intra_logits_mean = statistics.mean(train_epoch_logits["intra"])
            train_intra_logits_std = statistics.pstdev(train_epoch_logits["intra"])
            train_inter_logits_mean = statistics.mean(train_epoch_logits["inter"])
            train_inter_logits_std = statistics.pstdev(train_epoch_logits["inter"])
            train_max_probs_mean = statistics.mean(train_epoch_metrics["max_probs"])
            train_max_probs_std = statistics.pstdev(train_epoch_metrics["max_probs"])
            train_entropies_mean = statistics.mean(train_epoch_metrics["entropies"])
            train_entropies_std = statistics.pstdev(train_epoch_metrics["entropies"])
            valid_intra_logits_mean = statistics.mean(valid_epoch_logits["intra"])
            valid_intra_logits_std = statistics.pstdev(valid_epoch_logits["intra"])
            valid_inter_logits_mean = statistics.mean(valid_epoch_logits["inter"])
            valid_inter_logits_std = statistics.pstdev(valid_epoch_logits["inter"])
            valid_max_probs_mean = statistics.mean(valid_epoch_metrics["max_probs"])
            valid_max_probs_std = statistics.pstdev(valid_epoch_metrics["max_probs"])
            valid_entropies_mean = statistics.mean(valid_epoch_metrics["entropies"])
            valid_entropies_std = statistics.pstdev(valid_epoch_metrics["entropies"])

            print("\n####################################################")
            print("TRAIN MAX PROB MEAN:\t", train_max_probs_mean)
            print("TRAIN MAX PROB STD:\t", train_max_probs_std)
            print("VALID MAX PROB MEAN:\t", valid_max_probs_mean)
            print("VALID MAX PROB STD:\t", valid_max_probs_std)
            print("####################################################\n")

            print("\n####################################################")
            print("TRAIN ENTROPY MEAN:\t", train_entropies_mean)
            print("TRAIN ENTROPY STD:\t", train_entropies_std)
            print("VALID ENTROPY MEAN:\t", valid_entropies_mean)
            print("VALID ENTROPY STD:\t", valid_entropies_std)
            print("####################################################\n")

            with open(self.args.executions_raw_results_file_path, "a") as raw_results:
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "LOSS", train_loss))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "ACC1", train_acc1))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "SCALE", train_scale))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "INTRA_LOGITS MEAN", train_intra_logits_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "INTRA_LOGITS STD", train_intra_logits_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "INTER_LOGITS MEAN", train_inter_logits_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "INTER_LOGITS STD", train_inter_logits_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "MAX_PROBS MEAN", train_max_probs_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "MAX_PROBS STD", train_max_probs_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "ENTROPIES MEAN", train_entropies_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "TRAIN", "ENTROPIES STD", train_entropies_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "LOSS", valid_loss))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "ACC1", valid_acc1))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "SCALE", valid_scale))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "INTRA_LOGITS MEAN", valid_intra_logits_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "INTRA_LOGITS STD", valid_intra_logits_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "INTER_LOGITS MEAN", valid_inter_logits_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "INTER_LOGITS STD", valid_inter_logits_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "MAX_PROBS MEAN", valid_max_probs_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "MAX_PROBS STD", valid_max_probs_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "ENTROPIES MEAN", valid_entropies_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset, self.args.model_name, self.args.loss, self.args.execution, self.epoch, "VALID", "ENTROPIES STD", valid_entropies_std))

            print()
            print("TRAIN ==>>\tIELM: {0:.8f}\tIELS: {1:.8f}\tIALM: {2:.8f}\tIALS: {3:.8f}".format(
                train_inter_logits_mean, train_inter_logits_std, train_intra_logits_mean, train_intra_logits_std))
            print("VALID ==>>\tIELM: {0:.8f}\tIELS: {1:.8f}\tIALM: {2:.8f}\tIALS: {3:.8f}".format(
                valid_inter_logits_mean, valid_inter_logits_std, valid_intra_logits_mean, valid_intra_logits_std))
            print()
            print("\nDATA:", self.args.dataset)
            print("MODEL:", self.args.model_name)
            print("LOSS:", self.args.loss, "\n")

            if valid_acc1 > best_model_results["VALID ACC1"]:
                print("!+NEW BEST MODEL VALID ACC1!")
                best_model_results = {
                    "DATA": self.args.dataset,
                    "MODEL": self.args.model_name,
                    "LOSS": self.args.loss,
                    "EXECUTION": self.args.execution,
                    "EPOCH": self.epoch,
                    "TRAIN LOSS": train_loss,
                    "TRAIN ACC1": train_acc1,
                    "TRAIN SCALE": train_scale,
                    "TRAIN INTRA_LOGITS MEAN": train_intra_logits_mean,
                    "TRAIN INTRA_LOGITS STD": train_intra_logits_std,
                    "TRAIN INTER_LOGITS MEAN": train_inter_logits_mean,
                    "TRAIN INTER_LOGITS STD": train_inter_logits_std,
                    "TRAIN MAX_PROBS MEAN": train_max_probs_mean,
                    "TRAIN MAX_PROBS STD": train_max_probs_std,
                    "TRAIN ENTROPIES MEAN": train_entropies_mean,
                    "TRAIN ENTROPIES STD": train_entropies_std,
                    "VALID LOSS": valid_loss,
                    "VALID ACC1": valid_acc1,
                    "VALID SCALE": valid_scale,
                    "VALID INTRA_LOGITS MEAN": valid_intra_logits_mean,
                    "VALID INTRA_LOGITS STD": valid_intra_logits_std,
                    "VALID INTER_LOGITS MEAN": valid_inter_logits_mean,
                    "VALID INTER_LOGITS STD": valid_inter_logits_std,
                    "VALID MAX_PROBS MEAN": valid_max_probs_mean,
                    "VALID MAX_PROBS STD": valid_max_probs_std,
                    "VALID ENTROPIES MEAN": valid_entropies_mean,
                    "VALID ENTROPIES STD": valid_entropies_std,}

                print("!+NEW BEST MODEL VALID ACC1:\t\t{0:.4f} IN EPOCH {1}! SAVING {2}\n".format(valid_acc1, self.epoch, self.args.best_model_file_path))
                torch.save(self.model.state_dict(), self.args.best_model_file_path)
                np.save(os.path.join(self.args.experiment_path, "best_model"+str(self.args.execution)+"_train_epoch_logits.npy"), train_epoch_logits)
                np.save(os.path.join(self.args.experiment_path, "best_model"+str(self.args.execution)+"_train_epoch_metrics.npy"), train_epoch_metrics)
                np.save(os.path.join(self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_logits.npy"), valid_epoch_logits)
                np.save(os.path.join(self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_metrics.npy"), valid_epoch_metrics)

            print('!$$$$ BEST MODEL TRAIN ACC1:\t\t{0:.4f}'.format(best_model_results["TRAIN ACC1"]))
            print('!$$$$ BEST MODEL VALID ACC1:\t\t{0:.4f}'.format(best_model_results["VALID ACC1"]))

        with open(self.args.executions_best_results_file_path, "a") as best_results:
            best_results.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                best_model_results["DATA"],
                best_model_results["MODEL"],
                best_model_results["LOSS"],
                best_model_results["EXECUTION"],
                best_model_results["EPOCH"],
                best_model_results["TRAIN LOSS"],
                best_model_results["TRAIN ACC1"],
                best_model_results["TRAIN SCALE"],
                best_model_results["TRAIN INTRA_LOGITS MEAN"],
                best_model_results["TRAIN INTRA_LOGITS STD"],
                best_model_results["TRAIN INTER_LOGITS MEAN"],
                best_model_results["TRAIN INTER_LOGITS STD"],
                best_model_results["TRAIN MAX_PROBS MEAN"],
                best_model_results["TRAIN MAX_PROBS STD"],
                best_model_results["TRAIN ENTROPIES MEAN"],
                best_model_results["TRAIN ENTROPIES STD"],
                best_model_results["VALID LOSS"],
                best_model_results["VALID ACC1"],
                best_model_results["VALID SCALE"],
                best_model_results["VALID INTRA_LOGITS MEAN"],
                best_model_results["VALID INTRA_LOGITS STD"],
                best_model_results["VALID INTER_LOGITS MEAN"],
                best_model_results["VALID INTER_LOGITS STD"],
                best_model_results["VALID MAX_PROBS MEAN"],
                best_model_results["VALID MAX_PROBS STD"],
                best_model_results["VALID ENTROPIES MEAN"],
                best_model_results["VALID ENTROPIES STD"],))


    def train_epoch(self):
        print()
        self.model.train()

        loss_meter = utils.MeanMeter()
        accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        epoch_logits = {"intra": [], "inter": []}
        epoch_metrics = {"max_probs": [], "entropies": [], "max_logits": [], "mean_logits": []}

        for batch_index, batch_data in enumerate(self.trainset_loader_for_train):
            batch_index += 1

            inputs = batch_data[0]
            targets = batch_data[1]
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)

            if self.args.loss.split("_")[0] in ["dismax"]:
                inputs, targets = self.criterion.preprocess(inputs, targets)                      

            outputs = self.model(inputs)
            loss, scale, inter_logits, intra_logits = self.criterion(outputs, targets, debug=True)

            max_logits = outputs.max(dim=1)[0]
            mean_logits = outputs.mean(dim=1)
            probabilities = torch.nn.Softmax(dim=1)(outputs)
            max_probs = probabilities.max(dim=1)[0]
            entropies = utils.entropies_from_probabilities(probabilities)

            loss_meter.add(loss.item(), targets.size(0))
            accuracy_meter.add(outputs.detach(), targets.detach())

            intra_logits = intra_logits.tolist()
            inter_logits = inter_logits.tolist()
            if self.args.number_of_model_classes > 100:
                epoch_logits["intra"] = intra_logits
                epoch_logits["inter"] = inter_logits
            else:
                epoch_logits["intra"] += intra_logits
                epoch_logits["inter"] += inter_logits
            epoch_metrics["max_probs"] += max_probs.tolist()
            epoch_metrics["max_logits"] += max_logits.tolist()
            epoch_metrics["mean_logits"] += mean_logits.tolist()
            epoch_metrics["entropies"] += (entropies/math.log(self.args.number_of_model_classes)).tolist()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % self.args.print_freq == 0:
                print('Train Epoch: [{0}][{1:3}/{2}]\t'
                      'Loss {loss:.8f}\t\t'
                      'Acc1 {acc1_meter:.2f}\t'
                      'IELM {inter_logits_mean:.4f}\t'
                      'IELS {inter_logits_std:.8f}\t\t'
                      'IALM {intra_logits_mean:.4f}\t'
                      'IALS {intra_logits_std:.8f}'
                      .format(self.epoch, batch_index, len(self.trainset_loader_for_train),
                              loss=loss_meter.avg,
                              acc1_meter=accuracy_meter.value()[0],
                              inter_logits_mean=statistics.mean(inter_logits),
                              inter_logits_std=statistics.stdev(inter_logits),
                              intra_logits_mean=statistics.mean(intra_logits),
                              intra_logits_std=statistics.stdev(intra_logits)))

        print('\n#### TRAIN ACC1:\t{0:.4f}\n'.format(accuracy_meter.value()[0]))
        return loss_meter.avg, accuracy_meter.value()[0], scale, epoch_logits, epoch_metrics

    def validate_epoch(self, precompute_thresholds=True):
        print()
        self.model.eval()

        loss_meter = utils.MeanMeter()
        accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        epoch_logits = {"intra": [], "inter": []}
        epoch_metrics = {"max_probs": [], "entropies": [], "max_logits": [], "mean_logits": []}

        with torch.no_grad():
            for batch_index, batch_data in enumerate(self.valset_loader):
                batch_index += 1

                inputs = batch_data[0]
                targets = batch_data[1]
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)

                outputs = self.model(inputs)
                if self.args.loss.split("_")[0] in ["dismax"]:
                    loss, scale, inter_logits, intra_logits = self.criterion(outputs, targets, debug=True, precompute_thresholds=precompute_thresholds)
                else:
                    loss, scale, inter_logits, intra_logits = self.criterion(outputs, targets, debug=True)

                max_logits = outputs.max(dim=1)[0]
                mean_logits = outputs.mean(dim=1)
                probabilities = torch.nn.Softmax(dim=1)(outputs)
                max_probs = probabilities.max(dim=1)[0]
                entropies = utils.entropies_from_probabilities(probabilities)

                loss_meter.add(loss.item(), inputs.size(0))
                accuracy_meter.add(outputs.detach(), targets.detach())

                intra_logits = intra_logits.tolist()
                inter_logits = inter_logits.tolist()
                if self.args.number_of_model_classes > 100:
                    epoch_logits["intra"] = intra_logits
                    epoch_logits["inter"] = inter_logits
                else:
                    epoch_logits["intra"] += intra_logits
                    epoch_logits["inter"] += inter_logits
                epoch_metrics["max_probs"] += max_probs.tolist()
                epoch_metrics["max_logits"] += max_logits.tolist()
                epoch_metrics["mean_logits"] += mean_logits.tolist()
                epoch_metrics["entropies"] += (entropies/math.log(self.args.number_of_model_classes)).tolist()

                if batch_index % self.args.print_freq == 0:
                    print('Valid Epoch: [{0}][{1:3}/{2}]\t'
                          'Loss {loss:.8f}\t\t'
                          'Acc1 {acc1_meter:.2f}\t'
                          'IELM {inter_logits_mean:.4f}\t'
                          'IELS {inter_logits_std:.8f}\t\t'
                          'IALM {intra_logits_mean:.4f}\t'
                          'IALS {intra_logits_std:.8f}'
                          .format(self.epoch, batch_index, len(self.valset_loader),
                                  loss=loss_meter.avg,
                                  acc1_meter=accuracy_meter.value()[0],
                                  inter_logits_mean=statistics.mean(inter_logits),
                                  inter_logits_std=statistics.stdev(inter_logits),
                                  intra_logits_mean=statistics.mean(intra_logits),
                                  intra_logits_std=statistics.stdev(intra_logits)))

        print('\n#### VALID ACC1:\t{0:.4f}\n'.format(accuracy_meter.value()[0]))
        return loss_meter.avg, accuracy_meter.value()[0], scale, epoch_logits, epoch_metrics
