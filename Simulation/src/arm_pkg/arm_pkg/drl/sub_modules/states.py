import time
import serial
import struct
from sub_modules.configuration import Configuration

class States():
    def __init__(self):
        print("Initializing States module fiiisss")
        self.config = Configuration()

        time.sleep(0.3)

    def read_sensor_data(self):
        ser = serial.Serial(self.config.port1, baudrate=115200, timeout=1)
        try:
            return self._receive_sensor_data(ser)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.ser.close()

    def _receive_sensor_data(self, ser):

        format_string = '!10i 2f 3f'  # 10 integers, 2 floats, 3 floats

        # Read the packed data from the serial port
        packed_data = ser.read(struct.calcsize(format_string))

        # Unpack the received data
        unpacked_data = struct.unpack(format_string, packed_data)

        # Convert the tuple to a list and return
        return list(unpacked_data)
