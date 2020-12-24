import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torchtext import data
from torchtext.vocab import Vectors, Vocab
from collections import Counter
import numpy as np
import sys
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import numpy as np 


CLS = '[CLS]'
UNK = '[UNK]'
SEP = '[SEP]'
PAD = '[PAD]'



def field_load_vocab(field, vocab_path, novector=False):
    if not novector:
        vector=Vectors(name=vocab_path)
        counter=Counter()
        with open(vocab_path,'r',encoding='utf-8') as file:
            # file head
            file.readline()
            q=file.readline()
            while q:
                q=q.strip()
                counter[q.split()[0]]=1
                q=file.readline()
        field.vocab=Vocab(counter,vectors=vector)
    else:
        counter=Counter()
        with open(vocab_path,'r',encoding='utf-8') as file:
            lines=file.readlines()
            sum=1000000
            for index,line in enumerate(lines):
                line=line.strip()
                # 构造Vocab时，词频大的词排在词表的前面
                counter[line]=sum-index
        field.vocab=Vocab(counter,specials=[])


def train_one_epoch(model, train_iter, optimizer, epoch, args):
    model.train()

    train_loss = 0
    train_acc = 0
    times = 0

    print_loss = 0
    print_acc = 0

    for batch in train_iter:
        text, ori_label = batch.text, batch.label
        label = torch.argmax(ori_label, dim=1)
        mask = text > 0

        model.zero_grad()
        loss, output = model(
            text,
            labels=label,
            attention_mask=mask,
        )

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
            sys.stdout.flush()

            print_loss = 0
            print_acc = 0

        loss.backward()
        optimizer.step()

    train_loss /= times
    train_acc /= times
    return train_loss, train_acc


def evaluate(model, eval_iter):
    model.eval()
    test_loss = 0
    test_acc = 0
    times = 0

    f1 = 0
    corr = 0
    for batch in eval_iter:
        text, ori_label = batch.text, batch.label
        label = torch.argmax(ori_label, dim = 1)
        mask = text > 0

        loss, output = model(
            text,
            labels=label,
            attention_mask=mask,
        )

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

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.4f} ({:.2f}%, Correlation:{:4f}), F-score:{:4f})\n'.format(
        test_loss, test_acc, 100. * test_acc, corr, f1))
    return test_loss, test_acc,corr,f1


if __name__=='__main__':
    test()
