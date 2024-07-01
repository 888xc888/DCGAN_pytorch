"""
DCGAN的配置文件
"""
import os
import torch


class DCGAN_Config:
    def __init__(self):
        # 输入图像参数
        self.image_size = 64  # 图像尺寸
        self.channels = 3  # 输入图像的通道数 (例如 RGB 图像)

        # 训练参数
        self.batch_size = 50
        self.learning_rate = 0.0002
        self.beta1 = 0.5  # Adam 优化器的 beta1 参数
        self.learning_rate = 0.0002  # 学习率
        self.num_epochs = 400  # 训练的轮数
        self.sample_interval = 5  # 训练过程中生成的图像间隔
        self.checkpoint_interval = 10  # 训练过程中保存模型间隔
        self.nz = 100  # 噪声维度

        # 路径设置
        self.current_file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
        self.current_dir_path = os.path.dirname(self.current_file_path)  # 获取当前文件所在目录的绝对路径
        self.real_data_dir = self.current_dir_path + '\\data\\real_data'  # 训练中的真实数据集目录
        self.generated_data_dir = self.current_dir_path + '\\data\\generated_data'  # 训练中生成数据保存目录
        self.test_data_dir = self.current_dir_path + '\\data\\test_data'  # 训练好的GAN模型生成测试图像保存目录
        self.save_model_dir = self.current_dir_path + '\\pt'  # 模型保存目录

        # 其他参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
