import serial
import struct
from sub_modules.configuration import Configuration

class MoveJoints:
    def __init__(self):
        print("Initializing Move Joints module eiaau let's movee")
        self.config = Configuration()

    def move_joints(self, actions):

        ser = serial.Serial(self.config.port1, baudrate=115200, timeout=1)

        if len(actions)%2 != 0:
            print("Error: there has to be an even amount of values")
            return

        try:
            self._send_list(ser, actions)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            ser.close()

    def _send_list(self, ser, values):
        ser.write(struct.pack('!10f', *values))
        print(f"Sent values to {self.config.port1}: {values}")
