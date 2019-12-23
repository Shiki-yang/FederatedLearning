#!/usr/bin/python
# -*- coding: UTF-8 -*-

import socket
import threading
import json
'''
# 建立一个服务端
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(('localhost',6999)) #绑定要监听的端口
server.listen(5) #开始监听 表示可以使用五个链接排队
print('listening...')
'''

def initial_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 6999))
    server.listen(5)
    print('listening')
    return server

def link_handler(conn, address):
    print('handle:', address)
    data = conn.recv(1024, socket.MSG_WAITALL).decode()
    total_data = data
    num =  len(data)
    while len(data)>0:
        data = conn.recv(1024, socket.MSG_WAITALL).decode()
        print('len:', len(data))
        total_data += data
        num += len(data)
    #  print('total_data:', total_data)
    print('type:', type(json.loads(total_data)))
    print('num:', num)
    conn.close()
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
        t = threading.Thread(target=link_handler, args=(conn, address))
        t.start()

