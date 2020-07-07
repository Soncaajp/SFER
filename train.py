import os
import torch.utils.data
from torch import nn
from torch.nn import DataParallel
from datetime import datetime
from config import BATCH_SIZE, SAVE_FREQ, RESUME, SAVE_DIR, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU
from core import model_new
from core.utils import init_log
from dataloader.Affectnet_loader import Affectnet_aligned
from dataloader.Affectnet_test_loader import Affectnet_test_aligned
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def get_emotions(arr, num):
    l = []
    for i in arr:
        l.append(i[num])
    return torch.Tensor(l)

def aug(p=0.5):
    return transforms.Compose([
    transforms.ToPILImage(), 
    transforms.RandomHorizontalFlip(p = p),
    transforms.RandomGrayscale(),
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
])


writer = SummaryWriter()

# gpu init
gpu_list = ''
multi_gpus = False
if isinstance(GPU, int):
    gpu_list = str(GPU)
else:
    multi_gpus = True
    for i, gpu_id in enumerate(GPU):
        gpu_list += str(gpu_id)
        if i != len(GPU) - 1:
            gpu_list += ','
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

# other init
start_epoch = 1
save_dir = os.path.join(SAVE_DIR, MODEL_PRE + 'v2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info


# define trainloader and testloader

print('Trainloader')
trainset = Affectnet_aligned()#transform = aug())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8, drop_last=False)
print('Testloader')
testdataset = Affectnet_test_aligned()
testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,
                                         shuffle=False, num_workers=8, drop_last=False)

# define model
print('Model')
net = model_new.MobileFacenet()
# ArcMargin = model.ArcMarginProduct(128, trainset.class_nums)

if RESUME:
    ckpt = torch.load(RESUME)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1


# define optimizers
ignored_params = list(map(id, net.dense1.parameters()))
prelu_params_id = []
prelu_params = []
for m in net.modules():
    if isinstance(m, nn.PReLU):
        ignored_params += list(map(id, m.parameters()))
        prelu_params += m.parameters()
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer_ft = optim.SGD([
    {'params': base_params, 'weight_decay': 4e-5},
    {'params': net.dense1.parameters(), 'weight_decay': 4e-4},
    {'params': prelu_params, 'weight_decay': 0.0}
], lr=0.1, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)


net = net.cuda()
if multi_gpus:
    net = DataParallel(net)

weights= [0.26373185113172853,
            0.47345915139731315,
            0.0896753104944664,
            0.04962980183302689,
            0.022465498657987616,
            0.013395467450035576,
            0.0876429190354418]
class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)


best_acc = 0.0
best_epoch = 0
index = 1
for epoch in range(start_epoch, TOTAL_EPOCH+1):
    exp_lr_scheduler.step()
    # train model
    print('Train Epoch: {}/{} ...'.format(epoch, TOTAL_EPOCH))
    net.train()

    train_total_loss = 0.0
    total = 0
    since = time.time()
    for data in trainloader:
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        optimizer_ft.zero_grad()

        raw_logits = net(img)
        # true_label = []
        # for i in label.cpu().detach():
        #     l = []
        #     for j in range(7):
        #         if i == j:
        #             l.append(1)
        #         else:
        #             l.append(0)
        #     true_label.append(l)
        # true_label = torch.LongTensor(true_label).cuda()
        # true_raw_logits = []
        # for i in raw_logits.cpu().detach().numpy():
        #     l = []
        #     for j in range(7):
        #         l.append(i[j])
        #     true_raw_logits.append(l)
        total_loss = criterion(raw_logits,label)
        total_loss.backward()
        optimizer_ft.step()
        train_total_loss += total_loss * batch_size
        total += batch_size
        writer.add_scalar('Loss/epoch', train_total_loss, index)
        index += 1
    writer.add_scalar('Loss/train', train_total_loss, epoch)


    train_total_loss = train_total_loss / total
    time_elapsed = time.time() - since
    loss_msg = '    total_loss: {:.4f} time: {:.0f}m {:.0f}s'\
        .format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
    print(loss_msg)

    # # test model on Affectnet validation set
    if epoch % TEST_FREQ == 0:
        net.eval()
        print('Test Epoch: {} ...'.format(epoch))
        ma = []
        for data in testloader:
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            raw_logits = net(img)

            # true_label = []
            # for i in label.cpu().detach():
            #     l = []
            #     for j in range(7):
            #         if i == j:
            #             l.append(1)
            #         else:
            #             l.append(0)
            #     true_label.append(l)
            # true_raw_logits = []
            # for i in raw_logits.cpu().detach().numpy():
            #     l = []
            #     for j in range(7):
            #         l.append(i[j])
            #     true_raw_logits.append(l)

            test_loss = criterion(raw_logits, label)

            pred = []
            for l in raw_logits.cpu().detach().numpy():
                m = max(l)
                for i in range(len(l)):
                    if l[i] == m:
                        pred.append(i)
                        break
            ma.append(accuracy_score(label.cpu().detach().numpy(), pred))
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', np.mean(ma), epoch)

    # save model
    if epoch % SAVE_FREQ == 0:
        msg = 'Saving checkpoint: {}'.format(epoch)
        print(msg)
        if multi_gpus:
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))
print('finishing training')
