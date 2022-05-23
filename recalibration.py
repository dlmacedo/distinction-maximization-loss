# Code reused from: https://github.com/Jonathan-Pearce/calibration_library
import torch
from torch import nn
import metrics
import numpy as np
import scipy.optimize as opt


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, lr):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        #self.cuda()
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = metrics.ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        self.model = self.model.cuda()

        with torch.no_grad():
            for input, label in valid_loader:
                #input = input
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
        logits = logits.cpu()
        labels = labels.cpu()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion.loss(logits.numpy(),labels.numpy(),15)
        print('Before temperature - NLL: %.8f, ECE: %f' % (before_temperature_nll, before_temperature_ece))

        def ece_eval(tempearature):
            loss = ece_criterion.loss(logits.numpy()/tempearature,labels.numpy(),15)
            return loss

        ##########################################################################################################
        temperature, ece, _ = opt.fmin_l_bfgs_b(ece_eval, np.array([1.0]), approx_grad=True, bounds=[(0.001,100)])
        ##########################################################################################################

        print("temperature ====>>>>:", temperature[0])
        print("ece ============>>>>:", ece)

        return ece, temperature[0]

    def get_metrics(self, valid_loader):
        """
        valid_loader (DataLoader): validation set loader
        """

        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = metrics.ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion.loss(self.temperature_scale(logits).detach().numpy(),labels.numpy(),15)
        print('Optimal temperature: %.8f' % self.temperature.item())
        print('After temperature - NLL: %.8f, ECE: %f' % (after_temperature_nll, after_temperature_ece))

        #return self
        return after_temperature_ece
