import torch
import torch.autograd as autograd
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import numpy as np 

def train_one_epoch(model, train_iter, criterion, optimizer, epoch, args):
    model.train()

    train_loss = 0
    train_acc = 0
    times = 0
   
    print_loss = 0
    print_acc = 0

    for batch in train_iter:
        optimizer.zero_grad()
        text, ori_label = batch.text, batch.label
        label = torch.argmax(ori_label, dim = 1)  #分类任务计算
        output = model(text)
        loss = criterion(output, label) #label用regression的形式和classification(one hot)的形式试一下吧
        #乘一个权重
        _, pred = torch.max(output.data, dim=1)
        acc = (pred == label).sum().item() / label.shape[0]

        train_loss += loss.item()
        train_acc += acc
        print_loss += loss.item()
        print_acc += acc
        times += 1

        if times % args.print_frequency == 0:
            print('Train Epoch: {}, Iteration:{}, loss: {:.6f}, acc: {:.6f}'.format(
                epoch, times, print_loss / args.print_frequency, print_acc / args.print_frequency))
            print_loss = 0
            print_acc = 0

        loss.backward()
        optimizer.step()

    train_loss /= times
    train_acc /= times
    return train_loss, train_acc


def evaluate(model, eval_iter, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    times = 0

    f1 = 0
    corr = 0

    for batch in eval_iter:
        text, ori_label = batch.text, batch.label #归一化的label用于计算cor
        label = torch.argmax(ori_label,dim=1)
        output = model(text)
        loss = criterion(output, label)
        
        _, pred = torch.max(output.data, dim=1)
        acc = (pred == label).sum().item() / label.shape[0]  # 平均值

        temp_corr = 0
        for i in range(output.shape[0]):
            temp_corr += pearsonr(output[i].cpu().detach().numpy(), ori_label[i].cpu().detach().numpy())[0]
        temp_corr /= output.shape[0]
        corr += temp_corr

        f1 += f1_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), np.arange(8),average='micro')

        test_loss += loss.item()
        test_acc += acc
        times += 1

    test_loss /= times
    test_acc /= times
    corr /= times
    f1 /= times
    

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.4f} ({:.2f}%), Correlation:{:4f}, F-score:{:4f})\n'.format(
        test_loss, test_acc, 100. * test_acc, corr, f1))
    return test_loss, test_acc,corr,f1
