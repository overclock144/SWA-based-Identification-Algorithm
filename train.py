import torch

from dataloader_c import dataloader
from dataset_c import dataset
from log import Log
from loss_function_c import loss_fn
from model_c import model
from optimizer_c import optim, scheduler
from parm import epochs, dev, batch_size

# model = torch.load('./resnet50_model_best.pth')

logs = {'train': Log(batch_size=batch_size, epochs=epochs, dataset=dataset, dataset_name='train', format=True),
        'test': Log(batch_size=batch_size, epochs=epochs, dataset=dataset, dataset_name='test', format=True)}
logs['train'].start_log()
for epoch in range(epochs):
    logs['train'].epoch_start()
    # 训练模式
    model.train()
    # 开始训练
    for i, data in enumerate(dataloader['train']):
        # 输出10次Iteration的结果
        logs['train'].iteration(10, i)
        # 数据处理
        image, target = data
        image, target = image.to(dev), target.to(dev)
        # forward
        output = model(image)
        # loss
        loss = loss_fn(output, target)
        # 清空梯度
        optim.zero_grad()
        # backward
        loss.backward()
        # 优化权重
        optim.step()
        # acc计算
        _, prediction = output.max(1)
        acc = torch.sum(prediction == target)  # 每Iteration正确数量
        logs['train'].calculate_acc(acc)
        # loss计算
        logs['train'].calculate_loss(loss)
        # 图形化日志
        logs['train'].output_writer(loss, acc)
    # 优化lr
    scheduler.step()
    # save model
    torch.save(model, f'./output/model_{epoch}.pth')
    # 输出epoch结果
    logs['train'].epoch_end()
    with torch.no_grad():
        # 测试
        logs['test'].epoch_start()
        model.eval()
        for i, data in enumerate(dataloader['test']):
            # 输出10次Iteration的结果
            logs['test'].iteration(10, i)
            # 数据处理
            image, target = data
            image, target = image.to(dev), target.to(dev)
            # forward
            output = model(image)
            # loss
            loss = loss_fn(output, target)
            # acc计算
            _, prediction = output.max(1)
            acc = torch.sum(prediction == target)  # Iteration正确数量
            logs['test'].calculate_acc(acc)
            # loss计算
            logs['test'].calculate_loss(loss)
            # 图形化日志
            logs['test'].output_writer(loss, acc)
        logs['test'].epoch_end()
logs['test'].end_log()