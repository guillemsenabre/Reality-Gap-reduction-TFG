import time
import serial
import struct

class MoveJoints:
    def __init__(self):
        print("Initializing Move Joints module eiaau let's movee")

        time.sleep(0.3)

    def move_joints(self, actions, port, number_motors=10):

        ser = serial.Serial(port, baudrate=115200, timeout=1)

        if len(actions)%2 != 0:
            print("Error: there has to be an even amount of values")
            return

        try:
            self._send_list(ser, actions, number_motors)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            ser.close()

    def _send_list(self, ser, values, port, number_motors):
        ser.write(struct.pack(f'!{number_motors}f', *values))
        print(f"Sent values to {port}: {values}")
