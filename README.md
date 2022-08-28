# **Visit "The Robust Deep Learning Library" (our newest work) to quickly use this loss and much more:**

# **[The Robust Deep Learning Library](https://github.com/dlmacedo/robust-deep-learning)**

---

<img align="center" src="assets/dismax.png" width="750">

# Distinction Maximization Loss (DisMax)

## Efficiently Improving Out-of-Distribution Detection and Uncertainty Estimation by Replacing the Loss and Calibrating

>>**We keep single network inference efficiency. No hyperparameter tuning. We need to train only once. SOTA.**

>>**Read the full paper: [Distinction Maximization Loss: Efficiently Improving Out-of-Distribution Detection and Uncertainty Estimation by Replacing the Loss and Calibrating](https://arxiv.org/abs/2205.05874).**

>> ## **Train on CIFAR10, CIFAR100, and ImageNet.**

## Results

### Dataset=ImageNet, Model=ResNet18, Near OOD=ImageNet-O, Far OOD=XXXXXXXXX 

| Loss [Score] | Class (ACC) | Near OOD (AUROC) | Far OOD (AUROC) |
|:---|:---:|:---:|:---:|
| Cross-Entropy [MPS] | 69.9 | 52.4 | 00.0 |
| DisMax [MMLES] | 69.6 | 75.8 | 00.0 |

### Dataset=CIFAR

<img align="center" src="assets/table.png" width="750">

# Use DisMax in your project!!!

## Replace the SoftMax loss with the DisMax loss changing few lines of code!

### Replace the model classifier last layer with the DisMax loss first part:

```python
class Model(nn.Module):
    def __init__(self):
    (...)
    #self.classifier = nn.Linear(num_features, num_classes)
    self.classifier = losses.DisMaxLossFirstPart(num_features, num_classes)
```

### Replace the criterion by the DisMax loss second part:

```python
model = Model()
#criterion = nn.CrossEntropyLoss()
criterion = losses.DisMaxLossSecondPart(model.classifier)
```

### Preprocess before forwarding in the training loop:

```python
# In the training loop, add the line of code below for preprocessing before forwarding.
inputs, targets = criterion.preprocess(inputs, targets) 
(...)
# The code below is preexistent. Just keep the following lines unchanged!
outputs = model(inputs)
loss = criterion(outputs, targets)
```

## Detect during inference:

```python
# Return the score values during inference.
scores = model.classifier.scores(outputs) 
```

## Run the example:

```
python example.py
```

# Code

## Software requirements

Much code reused from [deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector), [odin-pytorch](https://github.com/facebookresearch/odin), and [entropic-out-of-distribution-detection](https://github.com/dlmacedo/entropic-out-of-distribution-detection).

### Please, install all package requirments runing the command bellow:

```bash
pip install -r requirements.txt
```

## Preparing the data

### Please, move to the `data` directory and run all the prepare data bash scripts:

```bash
# Download and prepare out-of-distrbution data for CIFAR10 and CIFAR100 datasets.
./prepare_cifar.sh
# Download and prepare out-of-distrbution data for ImageNet.
./prepare_imagenet.sh
```

## Reproducing the experiments

### Train and evaluate the classification, uncertainty estimation, and out-of-distribution detection performances:

```bash
./run_cifar100_densenetbc100.sh*
./run_cifar100_resnet34.sh*
./run_cifar100_wideresnet2810.sh*
./run_cifar10_densenetbc100.sh*
./run_cifar10_resnet34.sh*
./run_cifar10_wideresnet2810.sh*
./run_imagenet1k_resnet18.sh*
```

## Analizing the results

### Print the experiment results:

```bash
./analize.sh
```

# Citation

Please, cite our papers if you use our loss in your works:

```bibtex
@article{DBLP:journals/corr/abs-2205-05874,
  author    = {David Mac{\^{e}}do and
               Cleber Zanchettin and
               Teresa Bernarda Ludermir},
  title     = {Distinction Maximization Loss:
  Efficiently Improving Out-of-Distribution Detection and Uncertainty Estimation
  Simply Replacing the Loss and Calibrating},
  journal   = {CoRR},
  volume    = {abs/2205.05874},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.05874},
  doi       = {10.48550/arXiv.2205.05874},
  eprinttype = {arXiv},
  eprint    = {2205.05874},
  timestamp = {Tue, 17 May 2022 17:31:03 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-05874.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@article{DBLP:journals/corr/abs-2208-03566,
  author    = {David Mac{\^{e}}do},
  title     = {Towards Robust Deep Learning using Entropic Losses},
  journal   = {CoRR},
  volume    = {abs/2208.03566},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2208.03566},
  doi       = {10.48550/arXiv.2208.03566},
  eprinttype = {arXiv},
  eprint    = {2208.03566},
  timestamp = {Wed, 10 Aug 2022 14:49:54 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2208-03566.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
