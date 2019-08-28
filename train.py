from __future__ import print_function
import argparse
from math import log10

from libs.GANet.modules.GANet import MyLoss2
import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from models.GANet_deep import GANet
import torch.nn.functional as F
from dataloader.data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2048, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
parser.add_argument('--val_list', type=str, default='./lists/sceneflow_test_select.list', help="validation list")
parser.add_argument('--save_path', type=str, default='./checkpoint/', help="location to save models")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")

opt = parser.parse_args()

print(opt)
if opt.model == 'GANet11':
    from models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from models.GANet_deep import GANet
else:
    raise Exception("No suitable model found ...")
    
cuda = opt.cuda
#cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_path, opt.training_list, [opt.crop_height, opt.crop_width], opt.left_right, opt.kitti, opt.kitti2015, opt.shift)
test_set = get_test_set(opt.data_path, opt.val_list, [576,960], opt.left_right, opt.kitti, opt.kitti2015)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, drop_last=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = GANet(opt.max_disp)

criterion = MyLoss2(thresh=3, alpha=2)
if cuda:
    model = torch.nn.DataParallel(model).cuda()
optimizer=optim.Adam(model.parameters(), lr=opt.lr,betas=(0.9,0.999))
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
#        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def train(epoch):
    epoch_loss = 0
    epoch_error0 = 0
    epoch_error1 = 0
    epoch_error2 = 0
    valid_iteration = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), Variable(batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        target=torch.squeeze(target,1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:
            optimizer.zero_grad()
            
            if opt.model == 'GANet11':
                disp1, disp2 = model(input1, input2)
                disp0 = (disp1 + disp2)/2.
                if opt.kitti or opt.kitti2015:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * criterion(disp2[mask], target[mask])
                else:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
            elif opt.model == 'GANet_deep':
                disp0, disp1, disp2 = model(input1, input2)
                if opt.kitti or opt.kitti2015:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  criterion(disp2[mask], target[mask])
                else:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
            else:
                raise Exception("No suitable model found ...")
                
            loss.backward()
            optimizer.step()
            error0 = torch.mean(torch.abs(disp0[mask] - target[mask])) 
            error1 = torch.mean(torch.abs(disp1[mask] - target[mask]))
            error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))

            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error0 += error0.item()
            epoch_error1 += error1.item()
            epoch_error2 += error2.item()      
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}, Error: ({:.4f} {:.4f} {:.4f})".format(epoch, iteration, len(training_data_loader), loss.item(), error0.item(), error1.item(), error2.item()))
            sys.stdout.flush()

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. Error: ({:.4f} {:.4f} {:.4f})".format(epoch, epoch_loss / valid_iteration,epoch_error0/valid_iteration,epoch_error1/valid_iteration,epoch_error2/valid_iteration))

def val():
    epoch_error2 = 0

    valid_iteration = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
        target = torch.squeeze(target, 1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid>0:
            with torch.no_grad():
                disp2 = model(input1,input2)
                error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
                valid_iteration += 1
                epoch_error2 += error2.item()      
                print("===> Test({}/{}): Error: ({:.4f})".format(iteration, len(testing_data_loader), error2.item()))

    print("===> Test: Avg. Error: ({:.4f})".format(epoch_error2 / valid_iteration))
    return epoch_error2 / valid_iteration

def save_checkpoint(save_path, epoch,state, is_best):
    filename = save_path + "_epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + '_best.pth')
    print("Checkpoint saved to {}".format(filename))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 400:
       lr = opt.lr
    else:
       lr = opt.lr*0.1
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    error=100
    for epoch in range(1, opt.nEpochs + 1):
#        if opt.kitti or opt.kitti2015:
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        is_best = False
#        loss=val()
#        if loss < error:
#            error=loss
#            is_best = True
        if opt.kitti or opt.kitti2015:
            if epoch%50 == 0 and epoch >= 300:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)
        else:
            if epoch>=8:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)


    save_checkpoint(opt.save_path, opt.nEpochs,{
            'epoch': opt.nEpochs,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)


