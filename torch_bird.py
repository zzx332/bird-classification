
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
from config import opt
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# from utils import progress_bar
#Check if gpu support is available
batch_size = 32
learning_rate = 0.00001
# num_epochs = 10

data_dir = '//msralab/ProjectData/ehealth02/v-zzx/train'
test_dir = '//msralab/ProjectData/ehealth02/v-zzx/test'
log_dir = '//msralab/ProjectData/ehealth02/v-zzx/log'

data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
train_data = datasets.ImageFolder(root=data_dir,transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True, num_workers=4)
test_data = datasets.ImageFolder(root=test_dir,transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False, num_workers=4)

cuda_avail = torch.cuda.is_available()
# 读取pytorch自带的resnet-50模型,因为使用了预训练模型，所以会自动下载模型参数
model = models.resnet50(pretrained=True)


# # 对于模型的每个权重，使其不进行反向传播，即固定参数
# for param in model.parameters():
#     param.requires_grad = False
# # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层fc
# for param in model.fc.parameters():
#     param.requires_grad = True

class_num = 200  # 假设要分类数目是200
channel_in = model.fc.in_features  # 获取fc层的输入通道数
model.fc = nn.Linear(channel_in, class_num)
if cuda_avail:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(
#                         filter(lambda p: p.requires_grad, model.parameters()),#重要的是这一句
#                         lr=learning_rate)

loss_fn = torch.nn.CrossEntropyLoss()
confusion_matrix = meter.ConfusionMeter(200)


def adjust_learning_rate(epoch):

    lr = 0.00001

    if epoch >40:
        lr = lr / 100
    elif epoch > 20:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def test():
    # opt.__parse(kwargs)
    model.eval()

    # test_acc = 0.0
    # test_loss = 0.0
    total = 0.0
    num_correct = 0
    for i, (images, labels) in tqdm(enumerate(test_loader)):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        # inputs, labels = Variable(inputs), Variable(labels)
        # Predict classes using images from the test set
        outputs = model(images)
        # _, prediction = torch.max(outputs.data, 1)
        confusion_matrix.add(outputs.detach().squeeze(), labels.type(torch.LongTensor))
        total += float(labels.size(0))
        # test_acc += prediction.eq(labels).sum().item()
        # test_acc = test_acc / total
        _, prediction = torch.max(outputs, 1)  # move the computation in GPU
        num_correct += float(prediction.eq(labels).sum().item())
        # torch.cuda.empty_cache()
    test_acc = num_correct / total
    # Compute the average acc and loss over all 10000 test images
    # test_acc = test_acc / (len(test_data))

    return confusion_matrix, test_acc


def train(**kwargs):
    # best_acc = 0.0
    total = 0
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)
    for epoch in range(opt.max_epoch):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        num_correct = 0
        since = time.time()
        confusion_matrix.reset()
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            # inputs, labels = Variable(images), Variable(labels)
            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()
            # Adjust parameters according to the computed gradients
            optimizer.step()

            # train_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += labels.size(0)
            # train_acc += predicted.eq(labels).sum().item()

            # progress_bar(i, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss / (i + 1), 100. * train_acc / total, train_acc, total))
            train_loss += float(loss.item()*labels.size(0))
            confusion_matrix.add(outputs.detach(), labels.detach())
            # _, prediction = torch.max(outputs.data, 1)
            # train_acc += prediction.eq(labels).sum().item()
            _, prediction = torch.max(outputs, 1)  # move the computation in GPU
            num_correct += float(prediction.eq(labels).sum().item())
            train_acc = float(num_correct)
            total += float(labels.size(0))
            if (i + 1)%opt.print_freq == 0:
                vis.plot('loss', train_loss/total)
                vis.plot('train_acc', train_acc/total)
            torch.cuda.empty_cache()
        # Call the learning rate adjustment function
        # adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 50000 training images
        adjust_learning_rate(epoch)
        train_acc = train_acc / (len(train_data))
        train_loss = train_loss / (len(train_data))

        # Evaluate on the test set
        test_cm, test_acc = test()
        vis.plot('test_acc',test_acc)
        # Save the model if the test acc is greater than our current best
        # if test_acc > best_acc:
        #     save_models(epoch)
        #     best_acc = test_acc

        # Print the metrics
        elips_time = time.time() - since
        # print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}, Time:{:.0f}s".format(epoch, train_acc, train_loss,
        #                                                                                    test_acc,elips_time))
        vis.log("epoch:{epoch},loss:{loss},train_cm:{train_cm},test_cm:{test_cm}".format(
                    epoch = epoch,loss = train_loss,val_cm = str(test_cm.value()),train_cm=str(confusion_matrix.value())))

if __name__ == "__main__":
    import fire
    fire.Fire()

