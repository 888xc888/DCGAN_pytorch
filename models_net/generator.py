"""
DCGAN Generator
"""
import torch
import torch.nn as nn


class DCGAN_Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(DCGAN_Generator, self).__init__()

        self.main = nn.Sequential(
            # 输入是Z，经过全连接层后得到 (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 过渡到 (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 过渡到 (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 过渡到 (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 最终输出 (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# 示例代码：如何实例化这个生成器
if __name__ == '__main__':
    nz = 100  # 潜在向量的大小
    ngf = 64  # 生成器特征图大小
    nc = 3  # 图像的通道数，3表示RGB图像

    netG = DCGAN_Generator(nz, ngf, nc)
    print(netG)

    # 创建一个随机噪声向量
    fixed_noise = torch.randn(1, nz, 1, 1)

    # 通过生成器生成图像
    fake_image = netG(fixed_noise)
    print(fake_image.shape)  # 输出应该为 torch.Size([1, 3, 64, 64])
