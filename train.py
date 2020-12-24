import torch
import torch.nn as nn
import argparse
import time
import numpy as np
from torchtext import data
from torchtext.vocab import Vectors, GloVe
import jieba
import os
from model import RNN,CNN,MLP,AttentionRNN
from solve import train_one_epoch, evaluate


parser = argparse.ArgumentParser()
# learning
parser.add_argument('--dropout',default=0.5,type=float)
parser.add_argument('--fix_length',default='512',type=int)
parser.add_argument('--model_type',default='RNN',type=str,help="model used, model list = ['RNN','CNN','MLP','AttRNN']")
parser.add_argument('--max_epoch', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size(default: 1')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float, metavar='W',
                    help='weight decay(default: 1e-4)', dest='weight_decay')
parser.add_argument('-print-frequency', type=int, default=10,
                    help='how many steps to wait before logging training status [default: 50]')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--cuda',default=None)
parser.add_argument('--name', default='name',help = 'name of the experiment')
parser.add_argument('--vector', type=str, default='sogou')
parser.add_argument('--min_freq',type=int,default=1)
parser.add_argument('--cell',default='GRU')

# device
# parser.add_argument('--device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: 0]')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable the gpu')

args = parser.parse_args()
#args.cuda = (not args.no_cuda) and torch.cuda.is_available()


def tokenizer(text):
    return text.split(' ')


def main():
    # load data
    fix_length = args.fix_length

    if args.model_type != 'RNN' : #batch first && fix_length
        LABEL = data.Field(sequential=False,use_vocab=False,batch_first=True, dtype=torch.float32)
        TEXT = data.Field(sequential=True,fix_length = fix_length, tokenize=tokenizer, batch_first=True) 

    else:
        LABEL = data.Field(sequential=False, use_vocab=False,dtype=torch.float32)
        TEXT = data.Field(sequential=True,tokenize=tokenizer)

    fields = {'label': ('label', LABEL), 'text': ('text', TEXT)}
    print("ready to load data")
    train, val, test = data.TabularDataset.splits(path='newdata', train='train.json', validation='val.json',
                                                  test='test.json', format='json', fields=fields)
    train_length=len(train.examples)
    val_length=len(val.examples)
    test_length=len(test.examples)
    print('train_length:{}\nval_length:{}\ntest_length:{}'.format(train_length,val_length,test_length))
    
    print("ready to build vocab")
    # build vocab
    print('vector = ', args.vector)
    if args.vector == 'sogou':
        cache = 'word2vec/sogou_vector_cache'
        vector_path = 'word2vec/sgns.sogou.word'
    elif args.vector == 'baidu':
        cache = 'word2vec/baidu_vector_cache'
        vector_path = 'word2vec/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    elif args.vector == 'mixed':
        cache= 'word2vec/mixed_vector_cache'
        vector_path = 'word2vec/sgns.merge.word'
    #cache = '.vector_cache'
    if not os.path.exists(cache):
        print('cache not found')
        os.mkdir(cache)
    else:
        print('found cache')

    
    TEXT.build_vocab(train, min_freq=args.min_freq,vectors=Vectors(name = vector_path,cache=cache))
    #TEXT.build_vocab(train, val, test, min_freq=5,vectors=Vectors(name = '../sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5',cache=cache))
    #TEXT.build_vocab(train, val, test, min_freq=1,vectors=Vectors(name = '../sgns.merge.word',cache=cache))
    print("finish build vocab")
    vocab = TEXT.vocab
    print('vocab len = ', len(vocab))


    # 迭代器　返回batch
    device = torch.device('cuda:{}'.format(args.cuda) if args.cuda != None else 'cpu')
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=args.batch_size, device=device, shuffle=True,sort_key=lambda x: len(x.text), sort_within_batch=False)
    # 迭代器　返回batch

    # define model
    dropout = args.dropout
    model = 0
    print('model = ', args.model_type)
    if args.model_type == 'CNN':
        model = CNN(
            vocab = TEXT.vocab,
            input_size = 300,
            fix_length = fix_length,
            dropout = dropout
        )
    elif args.model_type == 'MLP':
        model = MLP(
            vocab = TEXT.vocab,
            input_size = 300,
            fix_length = fix_length,
            dropout = dropout
        )
    elif args.model_type == 'RNN':
        model = RNN(
        vocab=TEXT.vocab,
        input_size=300,
        hidden_size=300,
        num_layers=1,
        fc_size=64,
        cell = args.cell,
        dropout=dropout
        )
    elif args.model_type == 'AttRNN':
        model = AttentionRNN(
            vocab=TEXT.vocab,
            input_size=300,
            hidden_size=300,
            num_layers=1,
            fc_size=64,
            cell = args.cell,
            dropout=dropout
        )
    print(type(model))
    

    #weight = torch.Tensor([0.177625, 0.052946,0.061913,0.420154, 0.15670,0.07686, 0.042271,0.011528])

    if args.cuda != None:
        print('to cuda! device = ', device)
        model.to(device)
        #weight = torch.Tensor([0.177625, 0.052946,0.061913,0.420154, 0.15670, 0.07686, 0.042271, 0.011528]).to(device)
    print('device = ', device)
        #model.to(device)
    # define loss function and  optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train & val  & test
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    if not args.evaluate:
            best_val_acc = 0.0
            test_loss, test_acc,test_corr,test_f1= evaluate(model, test_iter, criterion)
            for epoch in range(args.max_epoch):
                train_loss, train_acc = train_one_epoch(model, train_iter, criterion, optimizer, epoch, args)
                val_loss, val_acc,val_corr,val_f1= evaluate(model, val_iter, criterion)
                
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)           
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print('ready to save data')
                    save_path = 'model_save/{}'.format(args.model_type)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                        print('mkdir ',save_path)
                    else:
                        print('{} has existed'.format(save_path))
                    torch.save(model.state_dict(), '{}/parameter_{}.pkl'.format(save_path, args.name))
                

                print('Epoch {} result is: train_acc={}, train_loss={}, val_acc={}, val_loss={}'.format(epoch, train_acc,
                                                                                                        train_loss, val_acc,
                                                                                                        val_loss))

            test_loss, test_acc,test_corr,testf1= evaluate(model, test_iter, criterion)
            '''
            np.save('{}_{}_train_loss.npy'.format(args.model_type, args.name), train_loss_list)
            np.save('{}_{}_train_acc.npy'.format(args.model_type, args.name), train_acc_list)
            np.save('{}_{}_val_loss.npy'.format(args.model_type, args.name), val_loss_list)
            np.save('{}_{}_val_acc.npy'.format(args.model_type, args.name), val_acc_list)
            '''
    else:
        print('evaluate mode!')
        #model.load_state_dict(torch.load('net_parameters.pkl'))
        if args.cuda == None:
            model.load_state_dict(torch.load('model_save/{}/parameter_{}.pkl'.format(args.model_type, args.name), map_location=device))  
        else:
            model.load_state_dict(torch.load('model_save/{}/parameter_{}.pkl'.format(args.model_type, args.name)))
        test_loss, test_acc,test_corr, test_f1 = evaluate(model, test_iter, criterion)
        

if __name__ == '__main__':
    main()
