import torch.nn as nn
import  torch



#BLOCK1
class BLOCK(nn.Module):#返回维度batch_size,filters,seq_len
    def __init__(self,filters, kernal_size,feature_size,seq_len):#batch_size,feature_size,seq_len
        super(BLOCK, self).__init__()
        self.cov = nn.Sequential(nn.Conv1d(feature_size,filters,1,stride=1,padding=0),#batch_size,filters,seq_len
                      nn.ReLU(),
                      nn.LayerNorm((filters,seq_len)),
                      nn.Conv1d(filters,filters,kernal_size,stride=1,padding=kernal_size//2),#batch_size,filters,seq_len 使用深度可分离卷积加速
                      nn.ReLU(),
                      nn.LayerNorm((filters,seq_len)),
                      nn.Conv1d(filters,filters,1,stride=1,padding=0),#batch_size,filters,seq_len
                      nn.ReLU(),
                      nn.LayerNorm((filters,seq_len)))
        self.org = nn.Conv1d(feature_size,filters,kernel_size=1,stride=1,padding=0)#batch_size,filters,seq_len

    def forward(self,x):
        o = self.cov(x)
        x = self.org(x)
        x = x+o                                                         #跳层连接结构
        return x

class BLOCK2(nn.Module):#返回维度batch_size,filters//2
    def __init__(self,filters,kernal_size,feature_size,seq_len):
        super(BLOCK2, self).__init__()
        self.block = nn.Sequential(BLOCK(filters,kernal_size,feature_size,seq_len),#batch_size,filters,seq_len
                       nn.MaxPool1d(2),                                             #batch_size,filters//2,seq_len//2

                       BLOCK(filters//2,kernal_size,filters,seq_len//2),            #batch_size,filters//2,seq_len//2
                       nn.AdaptiveAvgPool1d(1))                                     #batch_size,filters//2

    def forward(self,x):
        x = self.block(x)
        return x

class Model(nn.Module):
    def __init__(self, filters, feature_size, seq_len, co_feature, num_classes=19):
        super(Model, self).__init__()
        self.block3 = BLOCK2(filters=filters, kernal_size=3, feature_size=feature_size, seq_len=seq_len)
        self.block5 = BLOCK2(filters=filters, kernal_size=5, feature_size=feature_size, seq_len=seq_len)
        self.block7 = BLOCK2(filters=filters, kernal_size=7, feature_size=feature_size, seq_len=seq_len)

        self.fc = nn.Sequential(nn.Linear(filters // 2 * 3 + co_feature, 512),
                                nn.Dropout(0.3),
                                nn.Linear(512, 128),
                                nn.Dropout(0.3),
                                nn.Linear(128, 19))

        self.LN = nn.LayerNorm(co_feature)

    def forward(self, x):
        x, _ = x

        _ = self.LN(_)

        x = torch.cat([self.block3(x), self.block5(x),  self.block7(x)], dim=1)
        x = x.squeeze()
        x = torch.cat([x, _], dim=1)
        x = self.fc(x)

        return x



