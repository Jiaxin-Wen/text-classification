import torch
import torch.nn as nn
import argparse
from torchtext import data
import os
import transformers
from utils_myBert import train_one_epoch, evaluate, field_load_vocab, CLS, UNK, PAD, SEP
from bert import BERT

parser = argparse.ArgumentParser()
# learning
parser.add_argument('--max_epoch', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size(default: 1')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float, metavar='W',
                    help='weight decay(default: 1e-4)', dest='weight_decay')
parser.add_argument('-print-frequency', type=int, default=10,
                    help='how many steps to wait before logging training status [default: 50]')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--model', default='rnn_cache', help='model cache')
parser.add_argument('--name',default=1)
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable the gpu')


args = parser.parse_args()

#从预训练词表加载tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('../chinese_wwm_ext_pytorch/vocab.txt')

def myTokenizer(text):
    t = ''.join(text.split(' '))
    t = tokenizer.tokenize(t)[:1024]#max
    #t.insert(0, CLS) #加入clk和sep
    #t.append(SEP)
    return t


def main():
    # load data
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float32)
    TEXT = data.Field(sequential=True, tokenize=myTokenizer, lower=False, batch_first=True, unk_token=UNK, pad_token=PAD)
    fields = {'label': ('label', LABEL), 'text': ('text', TEXT)}
    train, val, test = data.TabularDataset.splits(path='../../newdata', train='train.json', validation='val.json',
                                                  test='test.json',format='json', fields=fields)
    train_length = len(train.examples)
    val_length = len(val.examples)
    test_length = len(test.examples)
    print('train_length:{}\nval_length:{}\ntest_length:{}'.format(train_length, val_length, test_length))


    #加载与训练数据的词表
    vocab_path = '../chinese_wwm_ext_pytorch/vocab.txt'
    with open(vocab_path, 'r', encoding='utf-8') as file:
        original_vocab_size = len(file.readlines())
    field_load_vocab(TEXT, vocab_path, novector=True)
    vocab_size = len(TEXT.vocab.stoi)
    print('original vocab size:{}'.format(original_vocab_size))

    print('vocab size:{}'.format(vocab_size))
    assert vocab_size == original_vocab_size

    cls_id = TEXT.vocab.stoi[CLS]
    sep_id = TEXT.vocab.stoi[SEP]
    print('cls id = {}, sep_id = {}'.format(cls_id, sep_id)) 


    device = torch.device('cuda:{}'.format(args.cuda) if args.cuda else 'cpu')
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=args.batch_size, device=device, shuffle=True,sort_key=lambda x: len(x.text), sort_within_batch=False)
    
    # define model
    
    config = transformers.BertConfig.from_pretrained('../chinese_wwm_ext_pytorch/bert_config.json',output_attentions=False,output_hidden_states=False)
    baseBert = transformers.BertModel.from_pretrained(
        '../chinese_wwm_ext_pytorch/pytorch_model.bin',
        config=config
    )
    model = BERT(baseBert,args.batch_size, device)
    if args.cuda:
        model.to(device)

    # define loss function and  optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train & val  & test
    best_val_acc = 0.0
    print('begin to train!')
    val_loss, val_acc,val_corr,val_f1 = evaluate(model, val_iter , criterion)
    for epoch in range(args.max_epoch):
        train_loss, train_acc = train_one_epoch(model, train_iter, criterion,optimizer, epoch, args)
        val_loss, val_acc,val_corr,val_f1= evaluate(model, val_iter,criterion )
        '''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not os.path.exists(sample_method + '/'):
                os.makedirs(sample_method + '/')
            torch.save(model.state_dict(), '{}/net_parameters.pkl'.format(sample_method))
        '''
        print('Epoch {} result is: train_acc={}, train_loss={}, val_acc={}, val_loss={}'.format(epoch, train_acc,
                                                                                                train_loss, val_acc,
                                                                                                val_loss))

    test_loss, test_acc,test_corr, test_f1 = evaluate(model, test_iter, criterion)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f} ({:.2f}%)\n'.format(test_loss, test_acc, test_acc * 100))


if __name__ == '__main__':
    main()
