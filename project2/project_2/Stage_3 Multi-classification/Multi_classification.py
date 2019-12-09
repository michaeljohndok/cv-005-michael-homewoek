import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
from torch import optim
from torch.optim import lr_scheduler
import copy
from sklearn.metrics import hamming_loss
import numpy as np
import cv2
import random
from Multi_Network import *

import time

ROOT_DIR = '../Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Multi_train_annotation.csv'
VAL_ANNO = 'Multi_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']  # 0,1
SPECIES = ['rabbits', 'rats', 'chickens']  # 0,1,2


class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):

        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_species = int(self.file_info.iloc[idx]['species'])
        label_classes = int(self.file_info.iloc[idx]['classes'])

        if label_species == 0:
            label_multi = torch.FloatTensor([1, 0, 0, 1, 0])
        elif label_species == 1:
            label_multi = torch.FloatTensor([0, 1, 0, 1, 0])
        else:
            label_multi = torch.FloatTensor([0, 0, 1, 0, 1])

        sample = {'image': image, 'species': label_species, 'classes': label_classes, 'label_multi': label_multi}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample


train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       ])

val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor(),
                                     ])

train_dataset = MyDataset(root_dir=ROOT_DIR + TRAIN_DIR,
                          annotations_file=TRAIN_ANNO,
                          transform=train_transforms)

test_dataset = MyDataset(root_dir=ROOT_DIR + VAL_DIR,
                         annotations_file=VAL_ANNO,
                         transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, SPECIES[sample['species']], CLASSES[sample['classes']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_species = 0
            count = 0

            for idx, data in enumerate(data_loaders[phase]):
                # print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_species = data['species'].to(device)
                label_multi = data['label_multi'].to(device)
                optimizer.zero_grad()
                count = count + 1

                with torch.set_grad_enabled(phase == 'train'):
                    x_species = model(inputs)
                    x_species = x_species.view(-1, 5)

                    loss = criterion(x_species, label_multi)
                    x_preds = np.array([np.where(l > 0.5, 1, 0) for l in x_species])

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects_species += 1 - hamming_loss(label_multi, x_preds)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_species = corrects_species / count
            epoch_acc = epoch_acc_species

            Accuracy_list_species[phase].append(100 * epoch_acc_species)
            print('{} Loss: {:.4f}  Acc_species: {:.2%}'.format(phase, epoch_loss, epoch_acc_species))
            #             logger.info('{} Loss: {:.4f}  Acc_species: {:.2%}'.format(phase, epoch_loss,epoch_acc_species))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc_species
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val species Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best val species Acc: {:.2%}'.format(best_acc))
    return model, Loss_list, Accuracy_list_species

def train_task():
    num_epochs = input("输入训练迭代次数：")
    num_epochs = int(num_epochs)
    network = Net().to(device)

    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(network.parameters(), lr=0.01)

    criterion = nn.BCELoss() # BCELoss CrossEntropyLoss
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs

    model, Loss_list, Accuracy_list_species = train_model(network, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
    #
    x = range(0, num_epochs)
    y1 = Loss_list["val"]
    y2 = Loss_list["train"]

    plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
    plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
    plt.legend()
    plt.title('train and val loss vs. epoches')
    plt.ylabel('loss')
    plt.savefig("train and val loss vs epoches.jpg")
    plt.close('all') # 关闭图 0
    #
    y5 = Accuracy_list_species["train"]
    y6 = Accuracy_list_species["val"]
    plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
    plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
    plt.legend()
    plt.title('train and val Species acc vs. epoches')
    plt.ylabel('Species accuracy')
    plt.savefig("train and val Species acc vs epoches.jpg")
    plt.close('all')


def check_loss():
    image = cv2.imread('./train and val loss vs epoches.jpg', 1)
    B, G, R = cv2.split(image)
    img_rgb = cv2.merge((R, G, B))
    plt.imshow(img_rgb)
    plt.show()


def check_acc():
    image = cv2.imread('./train and val Species acc vs epoches.jpg', 1)
    B, G, R = cv2.split(image)
    img_rgb = cv2.merge((R, G, B))
    plt.imshow(img_rgb)
    plt.show()


def get_lables(x_species):
    sp_list = []
    SPECIES = ['rabbits', 'rats', 'chickens', 'Mammals', 'Birds']  # [0,1,2,3,4]

    mask = x_species.ge(0.5)

    for idx, item in enumerate(mask):
        if item:
            sp_list.append(SPECIES[idx])
    return sp_list


def visualize_model():
    state_dict = torch.load('best_model.pt')
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            i = i + 1
            check = random.randint(0, 9)
            if i % 10 == check:

                inputs = data['image']
                label_species = data['species'].to(device)
                classes = data['classes'].to(device)

                if label_species == 0:
                    label_mul = torch.autograd.Variable(torch.LongTensor([[1, 0, 0, 1, 0]]))
                elif label_species == 1:
                    label_mul = torch.autograd.Variable(torch.LongTensor([[0, 1, 0, 1, 0]]))
                else:
                    label_mul = torch.autograd.Variable(torch.LongTensor([[0, 0, 1, 0, 1]]))

                x_species = model(inputs.to(device))
                x_species = x_species.view(-1)
                label_mul = label_mul.view(-1)

                Sigmoid = nn.Sigmoid()
                x_species = Sigmoid(x_species)

                x_lables = get_lables(x_species)
                y_lables = get_lables(label_mul)

                x_lables = ' '.join(x_lables)
                y_lables = ' '.join(y_lables)

                x_species = torch.nonzero(x_species)
                x_species = x_species.view(-1)

                plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
                plt.title('predicted species: {}\n ground-truth species:{}'.format(x_lables, y_lables))
                plt.show()

    print('测试预览结束')
    return True;