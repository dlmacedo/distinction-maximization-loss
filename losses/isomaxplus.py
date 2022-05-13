import torch.nn as nn
import torch.nn.functional as F
import torch


class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()."""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        distances_from_normalized_vectors = torch.cdist(F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist")       
        isometric_distances = torch.abs(self.distance_scale) * distances_from_normalized_vectors
        logits = -isometric_distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature


class IsoMaxPlusLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, model_classifier, entropic_scale=10.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=False):
        ##############################################################################
        ##############################################################################
        """Probabilities and logarithms are calculated separately and sequentially."""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss."""
        ##############################################################################
        ##############################################################################
        num_classes = logits.size(1)
        targets_one_hot = torch.eye(num_classes)[targets].long().cuda()
        probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * logits)
        probabilities_at_targets = probabilities_for_training[range(logits.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()

        if not debug:
            return loss
        else:
            intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')].detach().cpu().numpy()
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')].detach().cpu().numpy()
            return loss, self.model_classifier.distance_scale.item(), inter_logits, intra_logits, 
