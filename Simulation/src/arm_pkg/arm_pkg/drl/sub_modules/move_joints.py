import serial

class MoveJoints():
    def __init__(self, port1, port2):
        self.ser1 = serial.Serial(port1, baudrate=115200, timeout=1)
        self.ser2 = serial.Serial(port2, baudrate=115200, timeout=1)

    def move_joints(self, actions):
        pass
