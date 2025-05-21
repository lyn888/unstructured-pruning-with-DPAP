import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from .submodules.sparse import ConvBlock
from .submodules.layers import conv1x1, conv3x3, create_mslif

__all__ = ['VGGSNN']


class VGGSNN(nn.Module):
    def __init__(self):
        super(VGGSNN, self).__init__()
        self.skip = ['conv1']

        self.conv1 = ConvBlock(conv3x3(2, 64), nn.BatchNorm2d(64), create_mslif(),
                               sparse_neurons=False, sparse_weights=True)
        self.conv2 = ConvBlock(conv3x3(64, 128), nn.BatchNorm2d(128), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv3 = ConvBlock(conv3x3(128, 256), nn.BatchNorm2d(256), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.conv4 = ConvBlock(conv3x3(256, 256), nn.BatchNorm2d(256), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv5 = ConvBlock(conv3x3(256, 512), nn.BatchNorm2d(512), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.conv6 = ConvBlock(conv3x3(512, 512), nn.BatchNorm2d(512), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv7 = ConvBlock(conv3x3(512, 512), nn.BatchNorm2d(512), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.conv8 = ConvBlock(conv3x3(512, 512), nn.BatchNorm2d(512), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.pool4 = nn.AvgPool2d(2, 2)

        W = int(48 / 2 / 2 / 2 / 2)

        self.classifier = ConvBlock(conv1x1(512 * W * W, 100), None, None,
                                    sparse_neurons=False, sparse_weights=True)
        self.boost = nn.AvgPool1d(10, 10)
        self.init_weight()


        # 定义卷积层和全连接层的索引
        #修改
        self.convlayer = [-1, 0, 1, 3, 4, 6, 7,9 ,10]
        self.fclayer = [12]  # fc1 和 fc2
        #imgsize：每层图像的大小
        # self.imgsize = [32, 32, 32, 16, 16, 16, 8, 8, 8]
        # self.imgsize = [48, 48,48,24, 24, 24,12,12, 12, 6,6, 6,3]
        self.imgsize = [48,48, 48, 24, 24, 24, 12, 12, 12, 6, 6, 6]
        self.size = [2,64,128, 256, 256,  512, 512, 512,512]  # 每层输出的特征图通道数
        #self.size = [64,128,128, 256, 256,256, 512,512, 512, 512,512,512]  # 每层输出的特征图通道数
        # self.size_pool = [3, 256, 256, 256, 256, 256, 256, 256,256]  # 池化后的特征图通道数
        # self.size_pool = [64,128,128, 256, 256,256, 512,512, 512, 512,512,512]  # 池化后的特征图通道数
        self.size_pool = [2,64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]  # 池化后的特征图通道数

        self.fcsize = [512 * 6 * 6]  # 全连接层的输入大小
        self.ctrace={} #用于存储卷积层的轨迹
        self.fctrace={}#用于存储全连接层的轨迹
        self.csum={}#卷积层的轨迹累加值
        self.fcsum={}
        self.delta = 0.5
        self.step=10

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def connects(self):
        conn, total = 0, 0
        with torch.no_grad():
            sparse = torch.ones(1, 2, 48, 48, device='cuda')
            dense = torch.ones(1, 2, 48, 48, device='cuda')
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv2.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool1(sparse), self.pool1(dense)
            c, t, sparse, dense = self.conv3.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv4.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool2(sparse), self.pool2(dense)
            c, t, sparse, dense = self.conv5.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv6.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool3(sparse), self.pool3(dense)
            c, t, sparse, dense = self.conv7.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv8.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool4(sparse), self.pool4(dense)
            sparse, dense = sparse.view(1, -1, 1, 1), dense.view(1, -1, 1, 1)
            c, t, sparse, dense = self.classifier.connects(sparse, dense)
            conn, total = conn + c, total + t
        return conn, total

    def calc_c(self):
        with torch.no_grad():
            x = torch.ones(1, 2, 48, 48, device='cuda')
            x = self.conv1.calc_c(x)
            x = self.conv2.calc_c(x, [self.conv1])
            x = self.pool1(x)
            x = self.conv3.calc_c(x, [self.conv2])
            x = self.conv4.calc_c(x, [self.conv3])
            x = self.pool2(x)
            x = self.conv5.calc_c(x, [self.conv4])
            x = self.conv6.calc_c(x, [self.conv5])
            x = self.pool3(x)
            x = self.conv7.calc_c(x, [self.conv6])
            x = self.conv8.calc_c(x, [self.conv7])
            x = self.pool4(x)
            x = x.view(1, -1, 1, 1)
            x = self.classifier.calc_c(x, [self.conv8])
            return

    # def forward(self, x: torch.Tensor):
    #     x = x.transpose(0, 1)
    #     #### [N, T, C, H, W] -> [T, N, C, H, W]
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = functional.seq_to_ann_forward(x, self.pool1)
    #     x = self.conv3(x)
    #     x = self.conv4(x)
    #     x = functional.seq_to_ann_forward(x, self.pool2)
    #     x = self.conv5(x)
    #     x = self.conv6(x)
    #     x = functional.seq_to_ann_forward(x, self.pool3)
    #     x = self.conv7(x)
    #     x = self.conv8(x)
    #     x = functional.seq_to_ann_forward(x, self.pool4)
    #     x = x.view(x.shape[0], x.shape[1], -1, 1, 1)
    #     x = self.classifier(x)
    #     x = x.flatten(2).unsqueeze(2)
    #     #### [T, N, L] -> [T, N, C=1, L]
    #     out = functional.seq_to_ann_forward(x, self.boost).squeeze(2)
    #     return out

    def forward(self, x: torch.Tensor):
        # 转换输入数据维度 [N, T, C, H, W] -> [T, N, C, H, W]
        x = x.transpose(0, 1)
        #### [T, N, C, H, W]

        # 初始化输出和脉冲记录
        sum_out = 0
        spikes = []





        # x1 = self.conv1(x)

        for i in range(self.step):
        # 处理每个时间步

            spikest = []

            # 第一层卷积
            spikest.append(x.detach())
            x1 = self.conv1(x)
            spikest.append(x1.detach())



            # 第二层卷积
            x1 = self.conv2(x1)
            spikest.append(x1.detach())

            # 第一层池化
            x1 = functional.seq_to_ann_forward(x1, self.pool1)
            spikest.append(x1.detach())

            # 第三层卷积
            x1 = self.conv3(x1)
            spikest.append(x1.detach())

            # 第四层卷积
            x1 = self.conv4(x1)
            spikest.append(x1.detach())

            # 第二层池化
            x1 = functional.seq_to_ann_forward(x1, self.pool2)
            spikest.append(x1.detach())

            # 第五层卷积
            x1 = self.conv5(x1)
            spikest.append(x1.detach())

            # 第六层卷积
            x1 = self.conv6(x1)
            spikest.append(x1.detach())

            # 第三层池化
            x1 = functional.seq_to_ann_forward(x1, self.pool3)
            spikest.append(x1.detach())

            # 第七层卷积
            x1 = self.conv7(x1)
            spikest.append(x1.detach())

            # 第八层卷积
            x1 = self.conv8(x1)
            spikest.append(x1.detach())

            # 第四层池化
            x1 = functional.seq_to_ann_forward(x1, self.pool4)
            spikest.append(x1.detach())

            # 记录当前时间步的脉冲
            spikes.append(spikest)

            # 全连接层
            x1 = x1.view(x1.shape[0], x1.shape[1], -1, 1, 1)  # [T, N, C, H, W] -> [T, N, CxHxW, 1, 1]
            x1 = self.classifier(x1)
            x1 = x1.flatten(2).unsqueeze(2)  # [T, N, CxHxW] -> [T, N, C=1, L]

            # 聚合时间步的输出
            sum_out += functional.seq_to_ann_forward(x1, self.boost).squeeze(2)
        # for i in range(self.step):

        # 返回平均输出和脉冲记录
        return sum_out / self.step, spikes