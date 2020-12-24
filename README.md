### 1. 文件结构

![image-20200531092115476](../../pictures/typora_pic/image-20200531092115476.png)

```
-- bert/　bert模型实现

　-- chinese_wwm_ext_pytorch是bert的预训练数据

    -- main.py utils.py 是head+tail做法的Bert实现

    -- myBert目录下是分段输入Bert的实现，如报告中所说，我因显存限制未能测试模型效果，因此报告中没有汇报
　　这一方法的性能，也没有提供模型参数，无需复现。

-- train.py model.py solve.py提供了RNN,MLP,CNN,RNN+Attention的实现

-- newdata目录下是数据，已处理完毕
-- model_save目录下是保存的模型数据，提交时会删除掉，需下载此目录，注意保持目录结构，不要新增嵌套。
	清华云盘下载链接: https://cloud.tsinghua.edu.cn/d/f885293c5c444630a084/
-- word2vec目录下是词向量，提交时会删除掉，需下载至此目录,注意保持目录结构，不要新增嵌套。
	清华云盘下载链接:https://cloud.tsinghua.edu.cn/d/491f0e4168fb433ba960/
```



### 2. 复现方式

-e开启evaluate mode, --cuda指定GPU

#### MLP

```
python train.py --model_type MLP --lr 0.0001 -b 128 --wd 0.001 --name mixed_vector --vector mixed --fix_length 1560 -e --cuda 3
```



#### RNN

```
python train.py --model_type RNN --lr 1e-3 -b 16 --wd 1e-5 --name 1 --vector sogou --min_freq 1 -e --cuda 3
```



#### RNN+Attention

##### biGRU+Attention

```
 python train.py --model_type AttRNN --lr 1e-3 -b 32 --wd 1e-4 --name gru_1e-3_att --vector sogou --min_freq 1 --max_epoch 100 -e --cuda 3
```

##### biLSTM+Attention

```
  python train.py --model_type AttRNN --lr 1e-4 -b 32 --wd 1e-4 --name lstm --vector sogou --min_freq 1 --cell LSTM   --max_epoch 100 -e --cuda 3
```



#### BERT

```
cd bert/

python main.py --lr 1e-6 -b 6 --wd 0.0005 --max_epoch 20 --name fix_np --cuda 1 -e
```

