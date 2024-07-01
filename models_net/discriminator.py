"""
DCGAN Discriminator
"""
import torch
import torch.nn as nn


class DCGAN_Discriminator(nn.Module):
    def __init__(self, image_size, num_channels):
        super(DCGAN_Discriminator, self).__init__()
        self.image_size = image_size
        self.num_channels = num_channels

        # 定义判别器的网络结构
        self.main = nn.Sequential(
            # 输入形状：(num_channels) x image_size x image_size

            # 第一个卷积层
            # 输入：num_channels x image_size x image_size
            # 输出：64 x (image_size/2) x (image_size/2)
            nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 第二个卷积层
            # 输入：64 x (image_size/2) x (image_size/2)
            # 输出：128 x (image_size/4) x (image_size/4)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 第三个卷积层
            # 输入：128 x (image_size/4) x (image_size/4)
            # 输出：256 x (image_size/8) x (image_size/8)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 第四个卷积层
            # 输入：256 x (image_size/8) x (image_size/8)
            # 输出：512 x (image_size/16) x (image_size/16)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 第五个卷积层
            # 输入：512 x (image_size/16) x (image_size/16)
            # 输出：1 x (1) x (1) —— 这里假设 image_size/16 是1，即假设 image_size 是 16 的倍数
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # 使用 Sigmoid 激活函数，使输出值范围在 [0,1] 之间，表示图像的真假概率
            nn.Sigmoid()
        )

    def forward(self, input):
        # 前向传播函数，将输入数据通过 main 定义好的网络进行传播
        return self.main(input)


# 示例代码，展示如何实例化该 Discriminator 类并打印网络结构
if __name__ == "__main__":
    image_size = 64  # 指定输入图像的大小
    num_channels = 3  # 指定输入图像的通道数（例如，对于RGB图像，num_channels 为3）
    netD = DCGAN_Discriminator(image_size, num_channels)
    print(netD)
