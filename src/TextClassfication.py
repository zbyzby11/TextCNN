"""
TextCNN应用于文本分类（情感分析）
"""
import torch
from torch import nn, optim
from data_processing import InputData
from model import Model


class TextCNN(object):
    def __init__(self,
                 training_times=500,
                 batch_size=100,
                 embedding_size=128,
                 lr=0.001,
                 max_length=300):
        """
        进行文本分类的类（情感分析）
        :param training_times: 训练次数
        :param batch_size: 每一批的数据大小
        :param embedding_size: embedding大小
        :param lr: 学习率
        :param max_length: 每句话最大的长度，没有达到就padding，达到就截断
        """
        self.batch_size = batch_size
        self.training_times = training_times
        self.lr = lr
        self.embedding_size = embedding_size
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        input_data = InputData('../data/corpus.txt', max_length=max_length)
        self.train_iter, self.val_iter, self.test_iter, self.voca_size = input_data.create_iter(
            split_ratio=[0.8, 0.1, 0.1], batch_size=self.batch_size)
        self.model = Model(self.voca_size, self.embedding_size, max_length).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.training_times):
            self.model.train()
            flag = True
            for index, input in enumerate(self.train_iter):
                # torchtext的Field中的text字段是句子向量
                # torchtext的Field中的label字段是标签向量
                x = torch.LongTensor(input.text).to(self.device)
                label = torch.LongTensor(input.label).to(self.device)
                output = self.model(x)
                loss = self.criterion(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if flag:
                    print('epoch: {}|| loss is: {}'.format(epoch, loss.item()))
                    flag = False
            self.model.eval()
            # 模型验证
            if epoch % 5 == 0:
                val_data = next(iter(self.val_iter))
                val_x = torch.LongTensor(val_data.text).to(self.device)
                real_label = val_data.label.data.numpy()
                pre_output = self.model(val_x)
                pre_y = torch.max(pre_output, dim=1)[1].data.cpu().numpy()
                acc = sum(pre_y == real_label) / len(real_label)
                print('acc on valid set is: ', acc)
        print('------------------')
        print('------------------')
        test_data = next(iter(self.test_iter))
        test_x = torch.LongTensor(test_data.text).to(self.device)
        real_label_test = test_data.label.data.numpy()
        out = self.model(test_x)
        y = torch.max(out, dim=1)[1].data.cpu().numpy()
        test_acc = sum(y == real_label_test) / len(real_label_test)
        print('acc on test set is:', test_acc)


if __name__ == '__main__':
    m = TextCNN(training_times=200,
                batch_size=100,
                embedding_size=128,
                lr=0.0001,
                max_length=500)
    m.train()
