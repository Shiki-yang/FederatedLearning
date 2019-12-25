#!/usr/bin/python
# -*- coding: UTF-8 -*-

import socket
import threading
import json
import numpy as np
'''
# 建立一个服务端
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(('localhost',6999)) #绑定要监听的端口
server.listen(5) #开始监听 表示可以使用五个链接排队
print('listening...')
'''
global_param = {} # 全局参数：dict
received_param = [] # 已接受的参数：list
linking_client =  [('localhost', 7000)] # 已建立的连接：list
max_worker =  2 # client数量：int

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
    print('listening')
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
    num =  len(data)
    while len(data)>0:
        data = conn.recv(1024, socket.MSG_WAITALL).decode()
        total_data += data
        num += len(data)
    print('num:', num)
    # conn.close()
    return json.loads(total_data)

def param_merge(nets):
    """
    :param nets: client的网络参数列表。list of dictionary，维度不限，key不限，可适用于任意多个网络和任意维度的网络。
    :return: 合并后的网络参数，dictionary
    """
    """
    应对不同机器的网络参数的key 名字不同的情况：默认网络的 key不同时顺序仍然形同，可一一对应。
    """
    net = {}
    # for key in list(nets[0].keys()):
    for i in range(len(nets[0].keys())):
        keys = [list(net.keys())[i] for net in nets]
        # net[key] = np.sum([np.array(net[key]) for net in nets], axis=0)
        # net[key] = net[key] / len(nets)
        net[keys[0]] = np.mean([np.array(nets[j][keys[j]]) for j in range(len(nets))], axis=0)
        net[keys[0]] = net[keys[0]].tolist()
    return net

def load_net(file_name):
    '''
    :param file_name: 文件名
    :return: 网络参数:dict
    '''
    file = open(file_name, 'r')
    js = file.read()
    dic = json.loads(js)
    file.close()
    return dic

def param_send(conn, data):
    '''
    参数发送
    :param client:
    :param data:
    :return:
    '''
    conn.sendall(json.dumps(data).encode('utf-8'))

'''
if __name__ == '__main__':
    server = initial_server()
    while True:
        if len(linking_client) < max_worker:
            conn, address = server.accept()  # 等待连接，此处自动阻塞
            t = threading.Thread(target=param_rec, args=(conn, address))
            t.start()
        # len(linking_client) = max_worker = 2 : 执行参数聚合
        global_param = param_merge(received_param)
        # 参数聚合完成之后，将参数发送到client(conn, address)中
        for i in range(max_worker):
            param_send(linking_client[i], global_param)
        # 清空连接列表和接受参数列表
        linking_client.clear()
        received_param.clear()
'''

if __name__ == '__main__':

    server = initial_server('localhost', 6999)

    while True:
        # step.3-初始化server 并监听端口6999
        conn, addr = server.accept()

        # step.4-接受来自client的参数
        param = param_rec(conn,  addr)
        conn.close()
        received_param.append(param)

        print(len(received_param))  # 输出接受参数字典列表元素数量

        # step.5-聚合参数，读入两个参数，模拟多client
        net1 = load_net("param1.md")
        net2 = load_net("param2.md")
        received_param.append(net1)
        received_param.append(net2)
        print(len(received_param))
        global_param = param_merge(received_param)
        received_param.clear()
        print(global_param.keys())

        # step.6-发送global param
        # client = initial_client('localhost', 7000)
        for client in linking_client:
            conn = initial_client(client[0], client[1])
            conn.sendall(json.dumps(global_param).encode('utf-8'))
            conn.close()
