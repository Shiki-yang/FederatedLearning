
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy
import socket
import json
import time
import six

#设置device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trained_param = {} # 训练参数:dict
received_param  = {} # 接受参数:dict

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# root  = /datapool/workspace/jiangshanyang/Federated-Learning-PyTorch/data/cifar/
def cifar10_prepare(root):
    '''
    处理cifar10数据集
    :param root: 路径
    :return: trainloader
    '''
    print('now, we are gonging to prepare for the data')
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root, train= True, download= False, transform= transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    '''
    testset = torchvision.datasets.CIFAR10(root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    '''
    return trainloader

#vgg16模型定义
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        # view
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        # softmax 1 * 1 * 1000


    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)
        return out

class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def initial_server(IP, port):
    '''
    初始化一个server
    :param IP: string, 'localhost'
    :param port: int, 6999
    :return: 一个server对象
    '''
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((IP, port))
    server.listen(2)
    print('client is listening...')
    return server

def initial_client(IP, port):
    '''
    初始化一个client
    :param IP: string, 'localhost'
    :param port: int, 6999
    :return: 一个client对象
    '''
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((IP, port))
    return client

def param_rec(conn, addr):
    '''
    服务器端接受参数
    :param conn:
    :param addr:
    :return: 返回client参数字典：dict
    '''
    print('handle:', addr)
    data = conn.recv(1024, socket.MSG_WAITALL).decode()
    total_data = data
    num = len(data)
    while len(data)>0:
        data = conn.recv(1024, socket.MSG_WAITALL).decode()
        # print('len:', len(data))
        total_data += data
        num += len(data)
    print('num:', num)
    return json.loads(total_data)

def param_load(net,parm):
    '''
    加载新参数
    :param net:
    :param parm:
    :return:
    '''
    temp=net.state_dict()
    for key1,key2 in six.moves.zip(temp,parm):
        temp[key1]=torch.tensor(parm[key2])
    net.load_state_dict(temp)
    return net

if __name__ == '__main__':

    # step.1-训练网络，获得网络参数字典
    # step.1.1-准备数据集
    trainloader = cifar10_prepare('/datapool/workspace/jiangshanyang/Federated-Learning-PyTorch/data/cifar/')
    # trainloader = cifar10_prepare('/Users/yangyangj/Documents/DataGroup/FL_project/FederatedLearning/cifar/')
    print('next training...')
    net = CNNCifar()
    net = net.to(device)
    print(net)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum=0.9, weight_decay=5e-4)
    server = initial_server('localhost', 7001)
    # for i in range(2):
    while True:
        # step.1.2-train
        print('begin net train...')
        for epoch in range(1):
            net.train()
            for batch_index, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        print('end net train...')
        for name, param in net.state_dict().items():
            trained_param[name] = param.cpu().detach().numpy().tolist()
        # print(trained_param.keys())

        # step.2.1-建立client socket，连接server
        client = initial_client('localhost', 6999)
        # step.2.2-将网络参数字典发送至server
        print('next send all param...')
        client.sendall(json.dumps(trained_param).encode('utf-8'))
        trained_param.clear()  # 清空
        client.close()
        print('end send')

        # step.7-接受server参数
        # 等待接受新的参数（client发送-server接受-server聚合-server发送-client接受）
        # server = initial_server('localhost', 7000)
        conn, addr = server.accept()  # 阻塞
        print('receiving data from server...')
        received_param = param_rec(conn= conn, addr= addr)
        conn.close()
        print('end receiving, receive param:', len(received_param))

        # step.8-更新参数，重新训练网络
        net = param_load(net, received_param)
        received_param.clear()  # 清空
        # print(net.state_dict().keys())


