import scipy.io
import torch

from dataset_c import dataset
from parm import dev

# 载入模型
model = torch.load('./output/model_10_notest.pth')
# scipy读取标签
data = scipy.io.loadmat('./stanford_cars/devkit/cars_meta.mat')
with torch.no_grad():
    # 启用测试模式
    model.eval()
    # 载入单张图片
    image, target = dataset['test'][1907]
    # reshape→batch size设置为1
    image = image.reshape(1, 3, 224, 224)
    # to(device)
    inputs = image.to(dev)
    # forward
    outputs = model(inputs)
    # 通过sigmoid函数归一化,找到概率最大的索引
    outputs = torch.sigmoid(outputs)
    # 取预测结果的最大值
    _, p = torch.max(outputs, 1)
    # 输出
    print('正确结果:', data['class_names'][0][target])
    print('预测结果:', data['class_names'][0][p], '预测概率:', outputs[0][p])
