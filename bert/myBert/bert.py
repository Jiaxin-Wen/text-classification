import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT(nn.Module):
    def __init__(self, baseBert, batch_size,device):
        super(BERT,self).__init__()
        self.baseBert = baseBert #加载的预训练BERT
        print(device)
        self.device = device
        self.batch_size =batch_size
        self.fc = nn.Linear(192,8)
        #self.fc = nn.Linear(768,8)
    def forward(self, x):
        text_len = x.shape[1]
        output_list = []
        step = 256-32 #
        length = 256
        start = -1*step
        end = -1
        flag = False
        while not flag and (start < text_len):  #128区间　64步长滑动
            if (start+step) >= text_len:
               start = text_len
            else:
               start = start+step
       
       	    if (start+length) >= text_len:
               end = text_len
            else:
               end = start+length

            if end == text_len: #标记本次为最后一次滑动
                flag = True
            
            temp_text = x[:,start:end] 
            self.cls_column = torch.LongTensor([[101] for i in range(x.shape[0])]).to(self.device)
            self.sep_column = torch.LongTensor([[102] for i in range(x.shape[0])]).to(self.device)
            temp_text = torch.cat([self.cls_column, temp_text, self.sep_column], dim = 1)

            #在一整个batch加入clk和sep的id
            mask = temp_text>0
            #print('temp_text.shape =',temp_text.shape)
            output = self.baseBert(
                temp_text,
                attention_mask=mask,
            )
            output = output[1] #output[1]是clk的输出 　维度是[batch_size, hidden_size]
            output_list.append(output)
        #print('list len = ', len(output_list))

        kernel_size = len(output_list)*4
        concat_output = torch.cat(output_list,1)
        concat_output = concat_output.unsqueeze(1)
        final_output = F.max_pool1d(concat_output, kernel_size=kernel_size) #max_pooling
        final_output = final_output.squeeze(1)
        x = self.fc(final_output)
        return x
