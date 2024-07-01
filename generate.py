"""
This is the code for generating the synthetic data.
"""
import os
import torch
from models_net import generator
from torchvision.utils import save_image
import config


# 加载预训练的生成器模型
def load_model(cfg):
    netG = generator.DCGAN_Generator().to(cfg.device)
    checkpoint_path = os.path.join(cfg.save_model_dir, 'Final_generator.pth')  # 你保存的模型权重文件名
    netG.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    netG.eval()
    return netG


def generate_image(cfg, netG, num_images=1):
    noise = torch.randn(num_images, 100, 1, 1, device=cfg.device)  # 假设随机噪声维度为100
    fake_images = netG(noise)
    for i in range(num_images):
        save_image(fake_images[i], os.path.join(cfg.test_data_dir, f'generated_image_{i}.png'))


if __name__ == '__main__':
    cfg = config.DCGAN_Config()
    netG = load_model(cfg)
    generate_image(cfg, netG, num_images=5)  # 生成5张图片
