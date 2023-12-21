import serial
import struct
from arm_pkg.drl.configuration import Configuration

class MoveJoints:
    def __init__(self):
        config = Configuration()
        self.ser = serial.Serial(config.port1, baudrate=115200, timeout=1)

    def move_joints(self, actions):
        if len(actions)%2 != 0:
            print("Error: there has to be an even amount of values")
            return

        try:
            self._send_list(self.ser, actions)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.ser.close()

    def _send_list(self, ser, values):
        ser.write(struct.pack('!10f', *values))
        print(f"Sent values to {ser.port}: {values}")
