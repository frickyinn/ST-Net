import os
from torchvision import datasets, transforms

from model import Trainer


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root_path = './data/spatialLIBD'

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = datasets.ImageFolder(os.path.join(root_path, 'train'), transform=train_transforms)
testset = datasets.ImageFolder(os.path.join(root_path, 'test'), transform=test_transforms)


trainer = Trainer(n_classes=7, lr=3e-4)
trainer.fit(trainset, testset, epochs=50, num_workers=8)
