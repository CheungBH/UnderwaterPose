import socket
import threading
from config import config

class server:
    def __init__(self):
        # 第一步：创建一个socket
        self.ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 第二步：绑定监听的地址和port,方法bind()仅仅接收一个tuple
        # self.ss.bind(('192.168.50.101', config.port))
        self.ss.bind(('147.8.19.136', config.port))

        # 第三步：调用listen（）方法开始监听port。传入的參数指定等待连接的最大数量
        self.ss.listen(10)
        # ss.send(b'hello World!')
        # 第四步：server程序通过一个永久循环来接收来自client，accept()会等待并返回一个client的连接
    def connect(self,signal,status):
        sock, addr = self.ss.accept()
        # 创建一个新线程来处理TCP链接
        if signal == 0 or 1 or 2:
            threading.Thread(target=self.tcplink, args=(sock, addr,signal,status)).start()


    def tcplink(self,sock,addr,signal,status):
        print('signal sending %s:%s'%addr)
        s = "0, " + status
        mess = bytes(s, encoding='utf8')
        if signal == 0:
            sock.send(b'0' )
        elif signal == 1:
            sock.send(b'1')
        elif signal == 2:
            sock.send(mess)
        # while True:
        #     date=sock.recv(1024)
        #     if not date or date.decode('utf-8')=='exit':
        #         break
        #     print(date.decode('utf-8'))
        print(' %s:%s signal closing。。。。'%addr)


    def tcplink_img(self,sock,addr,signal):
        print('signal sending %s:%s'%addr)
        if signal == 0:
            sock.send(b'0')
        elif signal == 1:
            sock.send(b'1')
        elif signal == 2:
            sock.send(b'2')
        print(' %s:%s signal closing。。。。'%addr)





