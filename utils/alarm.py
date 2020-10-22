import serial
import time
from queue import Queue
from threading import Thread

class alert:
    def __init__(self):
        self.ser = serial.Serial(
            port="/dev/ttyUSB0",
            baudrate=9600#,
        )
        print(self.ser.isOpen())
        self.red_on = [0xA0,0x01,0x01,0xA2]
        self.yellow_on = [0xA0,0x02,0x01,0xA3]
        self.green_on = [0xA0,0x03,0x01,0xA4]
        self.buzzer_on = [0xA0,0x04,0x01,0xA5]
        self.red_off = [0xA0,0x01,0x00,0xA1]
        self.yellow_off = [0xA0,0x02,0x00,0xA2]
        self.green_off = [0xA0,0x03,0x00,0xA3]
        self.buzzer_off = [0xA0,0x04,0x00,0xA4]

    def start(self,signall):
        t = Thread(target=self.alram(signall), args=())
        t.daemon = True
        t.start()
        return self

    def alram(self,signal):
        self.red() if signal == 2 else self.yellow() if  signal == 1 else self.green()


    def red(self):
        print("red alert")
        self.ser.write(serial.to_bytes(self.yellow_off))
        # self.ser.write(serial.to_bytes(self.green_off))
        self.ser.write(serial.to_bytes(self.red_on))
        time.sleep(1)

    def yellow(self):
        # self.ser.write(serial.to_bytes(self.green_off))
        print("yellow alert")
        self.ser.write(serial.to_bytes(self.red_off))
        self.ser.write(serial.to_bytes(self.yellow_on))
        time.sleep(1)

    def green(self):
        self.ser.write(serial.to_bytes(self.yellow_off))
        # self.ser.write(serial.to_bytes(self.red_off))
        self.ser.write(serial.to_bytes(self.green_on))
        time.sleep(1)

    def buzzer_on(self):
        self.ser.write(serial.to_bytes(self.buzzer_on))
        time.sleep(1)

    def buzzer_off(self):
        self.ser.write(serial.to_bytes(self.buzzer_off))
        time.sleep(1)

    # ser.close()

