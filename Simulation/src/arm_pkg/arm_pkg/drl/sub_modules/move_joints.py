import serial
import time

class MoveJoints:
    def __init__(self, port):
        self.ser = serial.Serial(port, baudrate=115200, timeout=1)

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
        values_str = ','.join(map(str, values))
        ser.write(values_str.encode())
        print(f"Sent values to {ser.port}: {values_str}")