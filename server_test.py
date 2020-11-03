import socket
import threading
from config import config
import time
import eventlet
from socket import error as SocketError

class server:
    def __init__(self,bind = None):
        # pass
        self.ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.ss.bind(('192.168.50.101', config.port))
        self.ss.bind(('147.8.19.136', config.port))
        self.ss.listen(1000)


    def connect(self,signal,status):
        mes_from_client = 3
        sock1, addr1 = self.ss.accept()
        sock2, addr2 = self.ss.accept()
        if signal == 0 or signal ==1 or signal ==2:
            threading.Thread(target=self.tcplink, args=(sock1, addr1,signal,status)).start()
            threading.Thread(target=self.tcplink, args=(sock2, addr2, signal, status)).start()
        else:
            threading.Thread(target=self.tcplink_img, args=(sock1, addr1, signal)).start()
            threading.Thread(target=self.tcplink_img, args=(sock2, addr2, signal)).start()

        try:
            if addr1 == ('147.8.19.1',50844):
                mes_from_client = sock1.recv(1).decode('utf-8')
            elif addr2 == ('147.8.19.1', 50844):
                mes_from_client = sock2.recv(1).decode('utf-8')
            print("received data is",mes_from_client)
        except OSError:
            pass
        return mes_from_client

    def tcplink(self,sock,addr,signal,status):
        print('signal sending')
        if signal == 0:
            sock.send(b'0' )
        elif signal == 1:
            sock.send(b'1')
        elif signal == 2:
            s = "2, " + str(status)
            mess = bytes(s, encoding='utf8')
            sock.send(mess)
        sock.close()

    def tcplink_img(self,sock,addr,signal):
        print('img sending ')
        sock.send(signal)
        sock.close()






