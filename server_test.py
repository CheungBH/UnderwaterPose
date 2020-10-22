import socket
import threading
from config import config
import eventlet

class server:
    def __init__(self,bind = None):
        # pass
        # 第一步：创建一个socket
        self.ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 第二步：绑定监听的地址和port,方法bind()仅仅接收一个tuple
        # self.ss.bind(('192.168.50.101', config.port))
        self.ss.bind(('147.8.19.136', config.port))
        # 第三步：调用listen（）方法开始监听port。传入的參数指定等待连接的最大数量
        self.ss.listen(10)
        # 第四步：server程序通过一个永久循环来接收来自client，accept()会等待并返回一个client的连接

    def connect(self,signal,status):

        # ss = eventlet.listen('147.8.19.136', family=socket.AF_UNIX, backlog=50)
        # pool = eventlet.GreenPool(10000)
        # sock, addr = ss.accept()
        # if signal == 0 or signal == 1 or signal == 2:
        #     pool.spawn_n(self.tcplink(sock, addr,signal,status),sock)
        # else:
        #     pool.spawn_n(self.tcplink_img(sock, addr,signal,status),sock)
        # while True:
        sock, addr = self.ss.accept()
        # 创建一个新线程来处理TCP链接
        if signal == 0 or signal ==1 or signal ==2:
            threading.Thread(target=self.tcplink, args=(sock, addr,signal,status)).start()
        else:
            threading.Thread(target=self.tcplink_img, args=(sock, addr, signal, status)).start()



    def tcplink(self,sock,addr,signal,status):
        print('signal sending %s:%s'%addr)
        if signal == 0:
            sock.send(b'0' )
        elif signal == 1:
            sock.send(b'1')
        elif signal == 2:
            s = "2, " + status
            mess = bytes(s, encoding='utf8')
            sock.send(mess)
        print(' %s:%s signal closing。。。。'%addr)
        # self.ss.close()

    def tcplink_img(self,sock,addr,signal,status):
        print('img sending %s:%s'%addr)
        sock.send(signal)
        print(' %s:%s img closing。。。。'%addr)
        # self.ss.close()






