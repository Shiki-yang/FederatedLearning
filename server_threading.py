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
linking_client =  [] # 已建立的连接：list
max_worker =  2 # client数量：int

def initial_server():
    '''
    初始化服务器
    :return: 返回一个socket server实例
    '''
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 6999))
    server.listen(5)
    print('listening')
    return server

def param_rec(conn, address):
    '''
    服务器端接受参数
    :param conn:
    :param address:
    :return: 返回client参数字典：dict
    '''
    print('handle:', address)
    data = conn.recv(1024, socket.MSG_WAITALL).decode()
    total_data = data
    num =  len(data)
    while len(data)>0:
        data = conn.recv(1024, socket.MSG_WAITALL).decode()
        print('len:', len(data))
        total_data += data
        num += len(data)
    # print('total_data:', total_data)
    # print('type:', type(json.loads(total_data)))
    # print('num:', num)
    # conn.close()
    received_param.append(json.loads(total_data))
    linking_client.append((conn, address))

def param_merge(nets):
    """
	:param nets: client的网络参数列表。list of dictionary，维度不限，key不限，可适用于任意多个网络和任意维度的网络
	:return: 合并后的网络参数，dictionary
	"""
    net = {}
    for key in list(nets[0].keys()):
        net[key] = np.sum([np.array(net[key]) for net in nets], axis=0)
        net[key] = net[key] / len(nets)
        net[key] = net[key].tolist()
    return net

def param_send(client, data):
    '''
    参数发送
    :param client:
    :param data:
    :return:
    '''
    client[0].sendall(json.dumps(data).encode('utf-8'))

'''
conn, address = server.accept()
link_handler(conn, address)
'''

if __name__ == '__main__':
    server = initial_server()
    while True:     # 一个死循环，不断的接受客户端发来的连接请求

        conn, address = server.accept()  # 等待连接，此处自动阻塞
        # 每当有新的连接过来，自动创建一个新的线程，
        # 并将连接对象和访问者的ip信息作为参数传递给线程的执行函数
        t = threading.Thread(target=param_rec, args=(conn, address))
        t.start()
        print(linking_client)
        '''
        if len(linking_client) < max_worker: #max_worker = 2
            conn, address = server.accept()
            linking_client.append((conn, address))
            data = param_rec(conn=conn, address=address)
            received_param.append(data)
        # len(linking_client) = max_worker = 2 : 执行参数聚合
        global_param = param_merge(received_param)
        # 参数聚合完成之后，将参数发送到client(conn, address)中
        for i in  range(max_worker):
            param_send(linking_client[i], global_param)
        '''


