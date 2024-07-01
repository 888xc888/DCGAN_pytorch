"""
DCGAN的训练脚本
"""
# 导入系统模块
import os
# 导入配置文件
import config

cfg = config.DCGAN_Config()
# 导入模型
from models_net import discriminator
from models_net import generator

# 导入torch相关模块
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 导入绘图模块
import matplotlib.pyplot as plt

# 实例化模型
netD = discriminator.DCGAN_Discriminator(image_size=cfg.image_size, num_channels=cfg.channels).to(cfg.device)
netG = generator.DCGAN_Generator(nz=cfg.nz, ngf=cfg.image_size, nc=cfg.channels).to(cfg.device)

# 定义损失函数
criterion = nn.BCELoss()

# 定义优化器
optimizerD = optim.Adam(netD.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(cfg.image_size),  # 调整图像大小
    transforms.CenterCrop(cfg.image_size),  # 裁剪图像到指定大小
    transforms.ToTensor(),  # 将图像转化为Tensor
    transforms.Normalize([0.5] * cfg.channels, [0.5] * cfg.channels)  # 归一化
])

# 加载数据集
dataset = datasets.ImageFolder(root=cfg.real_data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# 训练开始
G_losses = []
D_losses = []

# 设置动态绘图
plt.figure(figsize=(10, 5))  # 创建一个用于绘制图像的图形窗口，并设置其大小为宽10英寸、高5英寸。
plt.ion()  # 开启交互模式，这允许在执行代码后的过程中实时更新图形，而无需阻塞代码运行。
fig, ax = plt.subplots()
line_G, = ax.plot(G_losses, label="G")
line_D, = ax.plot(D_losses, label="D")
plt.title("Generator and Discriminator Loss During Training")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()  # 展示图例

# 开始训练
for epoch in range(cfg.num_epochs):
    for i, data in enumerate(dataloader, 0):
        # 更新判别器：最大化 log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()  # 梯度清零

        real = data[0].to(cfg.device)  # 将从训练数据集中获取的真实图像数据加载到指定设备（通常是GPU）
        batch_size = real.size(0)  # 获取当前批次的图像数量
        label = torch.full((batch_size,), 1., dtype=torch.float, device=cfg.device)  # 创建标签，所有标签都设置为1（因为这些是真实数据）。

        output = netD(real).view(-1)  # 将真实数据通过判别器 netD，并将输出展平为一维向量
        errD_real = criterion(output, label)  # 计算真实数据的损失（与标签为1比较）
        errD_real.backward()  # 对损失进行反向传播，以计算梯度
        D_x = output.mean().item()  # 计算判别器对真实数据的平均判别值，并保存为 D_x

        noise = torch.randn(batch_size, cfg.nz, 1, 1, device=cfg.device)  # 生成随机噪声，作为生成器 netG 的输入
        fake = netG(noise)  # 通过生成器生成假数据（这个假数据用了两遍，第一次用于计算判别器的损失，第二次用于计算生成器的损失）
        label.fill_(0.)  # 创建标签，所有标签都设置为0（因为这些是假数据）。
        output = netD(fake.detach()).view(-1)  # 将假数据通过判别器 netD，并将输出展平为一维向量
        errD_fake = criterion(output, label)  # 计算假数据的损失（与标签为0比较）
        errD_fake.backward()  # 对损失进行反向传播，以计算梯度
        D_G_z1 = output.mean().item()  # 计算判别器对假数据的平均判别值，并保存为 D_G_z1

        errD = errD_real + errD_fake  # 计算判别器的总损失（真实数据损失和生成数据损失之和）
        optimizerD.step()  # 更新判别器的参数

        # 更新生成器：最大化 log(D(G(z)))
        netG.zero_grad()  # 梯度清零
        label.fill_(1.)  # 将标签填充为1。在训练生成器时，假数据生成的标签被设为真实数据的标签，即1。这是因为生成器的目标是让判别器认为生成的数据是真实的。
        output = netD(fake).view(-1)  # 将假数据（fake，由生成器生成）输入到判别器（netD）中，并将判别器的输出展平为一维向量。fake是在之前的代码中由生成器通过随机噪声生成的。
        errG = criterion(output, label)  # 使用二元交叉熵损失函数（criterion）计算生成器的损失（errG），该损失通过比较判别器的输出与标签1进行计算。
        errG.backward()  # 对生成器的损失进行反向传播，以计算生成器的梯度。
        D_G_z2 = output.mean().item()  # 计算判别器对假数据的平均判别值，并将其保存到变量 D_G_z2 中。
        optimizerG.step()  # 更新生成器的参数。这是通过生成器的优化器（optimizerG）进行的，此时已经使用反向传播计算出了生成器的梯度。

        # 保存损失以备后续绘图
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 动态更新绘制损失
        line_G.set_ydata(G_losses)
        line_G.set_xdata(range(len(G_losses)))
        line_D.set_ydata(D_losses)
        line_D.set_xdata(range(len(D_losses)))

        ax.relim()
        ax.autoscale_view()  # 重新调整坐标轴的范围以包含新的数据点，并自动缩放视图以适应新的数据。
        plt.pause(0.01)  # 暂停0.01秒，以使绘图更新到屏幕，这是实现动态绘图的关键。

        # 输出训练日志
        if i % 50 == 0:
            print(f'[{epoch}/{cfg.num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

        # 每隔几个batch保存生成的图片
        if i % cfg.sample_interval == 0:
            save_image(fake.data[:25], os.path.join(cfg.generated_data_dir, f'{epoch}_{i}.png'), nrow=5, normalize=True)

    # 每隔几个epoch保存模型
    if epoch % cfg.checkpoint_interval == 0:
        torch.save(netG.state_dict(), os.path.join(cfg.save_model_dir, f'netG_{epoch}.pth'))
        torch.save(netD.state_dict(), os.path.join(cfg.save_model_dir, f'netD_{epoch}.pth'))

# 训练结束，保存最终的loss曲线图并关闭动态绘图
plt.ioff()
plt.savefig(os.path.join(cfg.current_dir_path, 'loss_trend.png'))
plt.show()
