import os
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random


class ImageLoader:
    def __init__(self, args):
        self.args = args
        self.mnist = False

        if args.dataset == "cifar10":
            self.normalize = transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,])
            self.inference_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,])
            self.dataset_path = "data/cifar10"
            self.trainset_for_train = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=self.train_transform)
            self.trainset_for_infer = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=self.inference_transform)
            self.val_set = torchvision.datasets.CIFAR10(root=self.dataset_path, train=False, download=True, transform=self.inference_transform)

        elif args.dataset == "cifar100":
            self.normalize = transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,])
            self.inference_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,])
            self.dataset_path = "data/cifar100"
            self.trainset_for_train = torchvision.datasets.CIFAR100(root=self.dataset_path, train=True, download=True, transform=self.train_transform)
            self.trainset_for_infer = torchvision.datasets.CIFAR100(root=self.dataset_path, train=True, download=True, transform=self.inference_transform)
            self.val_set = torchvision.datasets.CIFAR100(root=self.dataset_path, train=False, download=True, transform=self.inference_transform)

        elif args.dataset == "tinyimagenet":
            #self.normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
                ])
            self.inference_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
                ])
            self.dataset_path = "data/tiny-imagenet-200"
            self.train_path = os.path.join(self.dataset_path, 'train')
            self.val_path = os.path.join(self.dataset_path, 'val')
            self.trainset_for_train = ImageFolder(self.train_path, transform=self.train_transform)
            self.trainset_for_infer = ImageFolder(self.train_path, transform=self.inference_transform)
            self.val_set = ImageFolder(self.val_path, transform=self.inference_transform)
            #self.outlier_data = None

        elif args.dataset == "imagenet1k":
            # future entropic losses will have a loader and ask the user to not apply any crop or resize transformation before loader!!!
            # future entropic losses will have a loader and ask the user to not apply any crop or resize transformation before loader!!!
            # future entropic losses will have a loader and ask the user to not apply any crop or resize transformation before loader!!!
            # future entropic losses will have a loader and ask the user to not apply any crop or resize transformation before loader!!!
            # ====>>>> na verdade, do not need loader!!! Pede para not apply any crop or resize transformation e faz no criterion.preprocess!!!
            # ====>>>> na verdade, do not need loader!!! Pede para not apply any crop or resize transformation e faz no criterion.preprocess!!!
            # ====>>>> na verdade, do not need loader!!! Pede para not apply any crop or resize transformation e faz no criterion.preprocess!!!
            # ====>>>> na verdade, do not need loader!!! Pede para not apply any crop or resize transformation e faz no criterion.preprocess!!!
            # ====>>>> na verdade, do not need loader!!! Pede para not apply any crop or resize transformation e faz no criterion.preprocess!!!
            # ====>>>> na verdade, do not need loader!!! Pede para not apply any crop or resize transformation e faz no criterion.preprocess!!!
            # ====>>>> na verdade, do not need loader!!! Pede para not apply any crop or resize transformation e faz no criterion.preprocess!!!
            # ====>>>> na verdade, do not need loader!!! Pede para not apply any crop or resize transformation e faz no criterion.preprocess!!!
            # input_size, interpolation e DEFAULT_CROP_RATIO will be passed as parameter to dismax, but configured when loading the model or dataset!!!
            # input_size, interpolation e DEFAULT_CROP_RATIO will be passed as parameter to dismax, but configured when loading the model or dataset!!!
            # input_size, interpolation e DEFAULT_CROP_RATIO will be passed as parameter to dismax, but configured when loading the model or dataset!!!
            # input_size, interpolation e DEFAULT_CROP_RATIO will be passed as parameter to dismax, but configured when loading the model or dataset!!!
            self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            self.train_transform = transforms.Compose([
                #transforms.RandomResizedCrop(args.input_size, scale=(3/4, 1.0), ratio=(3/4, 4/3), interpolation=args.interpolation),
                transforms.RandomResizedCrop(args.input_size, ratio=(3/4, 4/3), interpolation=args.interpolation),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,])
            self.inference_transform = transforms.Compose([
                transforms.Resize(int(args.input_size / args.DEFAULT_CROP_RATIO)),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                self.normalize,])
            #self.dataset_path = "/mnt/ssd/imagenet1k/images"
            self.dataset_path = "/mnt/ssd/imagenet1k/imagenet1k"
            self.train_path = os.path.join(self.dataset_path, 'train')
            self.val_path = os.path.join(self.dataset_path, 'val')
            self.trainset_for_train = ImageFolder(self.train_path, transform=self.train_transform)
            self.trainset_for_infer = ImageFolder(self.train_path, transform=self.inference_transform)
            self.val_set = ImageFolder(self.val_path, transform=self.inference_transform)


    def get_loaders(self):

        """
        if self.args.dataset == "imagenet1kffcv":
            IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
            IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
            DEFAULT_CROP_RATIO = 0.875
            input_size = 224

            this_device = f'cuda:{self.args.gpu_id}'
            train_path = Path('/mnt/ssd/train_500_0.50_90.ffcv')
            assert train_path.is_file()
            val_path = Path('/mnt/ssd/val_500_0.50_90.ffcv')
            assert val_path.is_file()

            self.decoder = RandomResizedCropRGBImageDecoder((input_size, input_size), scale=(0.08, 1), ratio=(0.75, 4 / 3))
            image_pipeline: List[Operation] = [
                self.decoder,
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(torch.device(this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
            ]
            label_pipeline: List[Operation] = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(torch.device(this_device), non_blocking=True)
            ]

            distributed = False
            order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
            trainset_loader_for_train = Loader('/mnt/ssd/val_500_0.50_90.ffcv',
                            batch_size=self.args.batch_size,
                            num_workers=8,
                            order=order,
                            os_cache=False,
                            drop_last=True,
                            pipelines={
                                'image': image_pipeline,
                                'label': label_pipeline
                            },
                            distributed=distributed)

            cropper = CenterCropRGBImageDecoder((input_size, input_size), ratio=DEFAULT_CROP_RATIO)
            image_pipeline = [
                cropper,
                ToTensor(),
                ToDevice(torch.device(this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
            ]
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(torch.device(this_device),
                non_blocking=True)
            ]

            valset_loader = Loader('/mnt/ssd/val_500_0.50_90.ffcv',
                            batch_size=self.args.batch_size,
                            num_workers=8,
                            order=OrderOption.SEQUENTIAL,
                            drop_last=False,
                            pipelines={
                                'image': image_pipeline,
                                'label': label_pipeline
                            },
                            distributed=distributed)
            trainset_loader_for_infer = None

        else:
        """
        trainset_loader_for_train = DataLoader(
            self.trainset_for_train, batch_size=self.args.batch_size, shuffle=True, num_workers=8,
            worker_init_fn=lambda worker_id: random.seed(self.args.seed + worker_id))
        trainset_loader_for_infer = DataLoader(
            self.trainset_for_infer, batch_size=self.args.batch_size, shuffle=False, num_workers=8)
        valset_loader = DataLoader(
            self.val_set, batch_size=self.args.batch_size, shuffle=False, num_workers=8)

        return trainset_loader_for_train, trainset_loader_for_infer, valset_loader
