"""
TextCNN网络结构
"""
import torch
from torch.nn import functional as F
from torch import nn


class Model(nn.Module):
    def __init__(self, voca_size, embedding_size, max_length):
        super(Model, self).__init__()
        self.voca_size = voca_size
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.emb = nn.Embedding(self.voca_size, self.embedding_size)
        # 将一句话的embedding看做是一张图片，卷积核长度为embedding-size大小
        self.conv = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(5, self.embedding_size))
        # 全连接层的维度需要测试才能确定,这里维度依赖卷积核和max_pooling
        self.fc = nn.Linear(int(100 * (self.max_length - 5 + 1) / 4 * 1), 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 输入x = [batch_size, seq_length]
        x = x.unsqueeze(1)
        # 输入x = [batch_size, 1, seq_length] -> [batch_size, 1, seq_length, embedding_size]
        x = self.emb(x)
        # 因为卷积操作是对一句话embedding矩阵的seq_length做的操作，所以
        # 需要将seq_length维度放在这个矩阵的最后
        # x = [batch_size, seq_length, embedding_size] -> [batch_size, embedding_size, seq_length]
        # x = x.permute(0, 1, 3, 2)
        # 卷积操作
        # x = [batch_size, 1, seq_length, embedding_size] -> [batch_size, 100, seq_length-5+1, 1]
        x = self.conv(x)
        x = F.relu(x)
        # x = [batch_size, 100, seq_length-5+1, 1] -> [batch_size, 100, (seq_length-5+1) / 4, 1]
        x = F.max_pool2d(x, (4, 1))
        x = F.relu(x)
        # x = [batch_size, 100, (seq_length-5+1) / 4, 1] -> [batch_size, 100*(seq_length-5+1) / 4 * 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x

# c = Model(1200, 128)
# d = torch.randint(0, 1000, (4, 300)).long()
# x = c(d)
# print(x.shape)
