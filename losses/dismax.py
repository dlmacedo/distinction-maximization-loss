import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class DisMaxLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()."""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(DisMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.constant_(self.distance_scale, 1.0)
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)        
        self.validationset_available = nn.Parameter(torch.tensor([False]), requires_grad=False)
        self.precomputed_thresholds = nn.Parameter(torch.Tensor(2, 25), requires_grad=False)
        self.score_type = "MMLES"

    def forward(self, features):
        distances_from_normalized_vectors = torch.cdist(F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist") / math.sqrt(2.0)    
        isometric_distances = torch.abs(self.distance_scale) * distances_from_normalized_vectors
        logits = -(isometric_distances + isometric_distances.mean(dim=1, keepdim=True))
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature

    def scores(self, logits):
        if self.score_type == "MPS": # the maximum probability score
            probabilities = torch.nn.Softmax(dim=1)(logits)
            scores = probabilities.max(dim=1)[0]
        elif self.score_type == "ES": # the negative entropy score
            probabilities = torch.nn.Softmax(dim=1)(logits)
            scores = (probabilities * torch.log(probabilities)).sum(dim=1)
        elif self.score_type == "MDS": # the minimum distance score
            scores = logits.max(dim=1)[0]
        elif self.score_type == "MMLS": # the max-mean logit score
            scores = logits.max(dim=1)[0] + logits.mean(dim=1)
        elif self.score_type == "MMLES": # the max-mean logit entropy score
            probabilities = torch.nn.Softmax(dim=1)(logits)
            scores = logits.max(dim=1)[0] + logits.mean(dim=1) + (probabilities * torch.log(probabilities)).sum(dim=1)
        elif self.score_type == "MMLEPS": # the max-mean logit entropy probability score
            probabilities = torch.nn.Softmax(dim=1)(logits)
            scores = logits.max(dim=1)[0] + logits.mean(dim=1) + (probabilities * torch.log(probabilities)).sum(dim=1) + probabilities.max(dim=1)[0]
        else:
            sys.exit('You should use a valid score type!!!')
        return scores


class DisMaxLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, model_classifier, regularization="fractional_probability", batches_to_accumulate=32):
        super(DisMaxLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.regularization = regularization
        self.batches_to_accumulate = batches_to_accumulate
        self.entropic_scale = 10
        self.accumulated_batches = 0
        self.accumulated_scores = {"train": None, "validation": None}

    def preprocess(self, inputs, targets):
        if self.regularization == "fractional_probability":
            batch_size = inputs.size(0)
            half_batch_size = batch_size//2
            idx = torch.randperm(batch_size)
            inputs = inputs[idx].view(inputs.size())
            targets = targets[idx].view(targets.size())
            if len(inputs.size()) == 4:
                W = inputs.size(2)
                H = inputs.size(3)
                #print("Regularization: Fractional Probability")
                inputs[half_batch_size:, :, W//2:, :H//2] = torch.roll(inputs[half_batch_size:, :, W//2:, :H//2], 1, 0)
                inputs[half_batch_size:, :, :W//2, H//2:] = torch.roll(inputs[half_batch_size:, :, :W//2, H//2:], 2, 0)
                inputs[half_batch_size:, :, W//2:, H//2:] = torch.roll(inputs[half_batch_size:, :, W//2:, H//2:], 3, 0)
        return inputs, targets

    def forward(self, logits, targets, debug=False, precompute_thresholds=False):
        ##############################################################################
        ##############################################################################
        """Probabilities and logarithms are calculated separately and sequentially."""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss."""
        ##############################################################################
        ##############################################################################
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        half_batch_size = batch_size//2
        targets_one_hot = torch.eye(num_classes)[targets].long().cuda()

        if self.model_classifier.training:
            partition = "train"
            probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * logits)
            if self.regularization is None: # no regularization
                probabilities_at_targets = probabilities_for_training[range(batch_size), targets]
                loss = -torch.log(probabilities_at_targets).mean()
            else: # regularization                                
                probabilities_at_targets = probabilities_for_training[range(half_batch_size), targets[:half_batch_size]]
                loss_not_regularized = -torch.log(probabilities_at_targets).mean()
                if self.regularization == "fractional_probability":
                    #print("Regularization: Fractional Probability")
                    targets_one_hot_0 = torch.eye(num_classes)[torch.roll(targets[half_batch_size:], 0, 0)].long().cuda()
                    targets_one_hot_1 = torch.eye(num_classes)[torch.roll(targets[half_batch_size:], 1, 0)].long().cuda()
                    targets_one_hot_2 = torch.eye(num_classes)[torch.roll(targets[half_batch_size:], 2, 0)].long().cuda()
                    targets_one_hot_3 = torch.eye(num_classes)[torch.roll(targets[half_batch_size:], 3, 0)].long().cuda()
                    target_distributions_0 = torch.where(targets_one_hot_0 != 0, torch.tensor(0.25).cuda(), torch.tensor(0.0).cuda())
                    target_distributions_1 = torch.where(targets_one_hot_1 != 0, torch.tensor(0.25).cuda(), torch.tensor(0.0).cuda())
                    target_distributions_2 = torch.where(targets_one_hot_2 != 0, torch.tensor(0.25).cuda(), torch.tensor(0.0).cuda())
                    target_distributions_3 = torch.where(targets_one_hot_3 != 0, torch.tensor(0.25).cuda(), torch.tensor(0.0).cuda())
                    target_distributions_total = target_distributions_0 + target_distributions_1 + target_distributions_2 + target_distributions_3
                    loss_regularization = F.kl_div(torch.log(probabilities_for_training[half_batch_size:]), target_distributions_total, reduction='batchmean')
                else:
                    sys.exit('You should pass a valid regularization!!!')
                loss = (loss_not_regularized + loss_regularization)/2

        else: # validation
            partition = "validation"
            self.model_classifier.validationset_available[0] = True
            probabilities_for_inference = nn.Softmax(dim=1)(logits)
            probabilities_at_targets = probabilities_for_inference[range(batch_size), targets]
            loss = -torch.log(probabilities_at_targets).mean()

        if precompute_thresholds:
            scores = self.model_classifier.scores(logits).detach().cpu().numpy()
            if self.accumulated_scores[partition] is not None:
                self.accumulated_scores[partition] = np.concatenate((self.accumulated_scores[partition], scores), axis=0)
            else:
                self.accumulated_scores[partition] = scores
            self.accumulated_batches += 1
            if self.accumulated_batches == self.batches_to_accumulate:
                partition_index = 0 if partition == "train" else 1
                for index, percentile in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]):
                    self.model_classifier.precomputed_thresholds[partition_index, index] = np.percentile(
                        self.accumulated_scores[partition], percentile)
                print("In-Distribution-Based Precomputed Thresholds [Based on Train Set]:\n", self.model_classifier.precomputed_thresholds.data[0])
                if self.model_classifier.validationset_available.data.item():
                    print("In-Distribution-Based Precomputed Thresholds [Based on Valid Set]:\n", self.model_classifier.precomputed_thresholds.data[1])
                self.accumulated_scores[partition] = None
                self.accumulated_batches = 0

        if not debug:
            return loss
        else:
            intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')].detach().cpu().numpy()
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')].detach().cpu().numpy()
            return loss, self.model_classifier.distance_scale.item(), inter_logits, intra_logits, 
