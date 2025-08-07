# 给我写一个最简单的deeplearning代码
# 这是一个简单的深度学习代码示例，使用pytorch库来创建一个简单的神经网络模型。

import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 10)        # 隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入
        x = torch.relu(self.fc1(x))  # 激活函数
        x = self.fc2(x)  # 输出层
        return x
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型
model = SimpleNN().to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])                  
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')