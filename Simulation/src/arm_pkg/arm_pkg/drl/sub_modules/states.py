import time
import serial
import struct
from sub_modules.configuration import Configuration

class States():
    def __init__(self):
        print("Initializing States module fiiisss")
        self.config = Configuration()
        self.ser = None

        time.sleep(0.3)

    def read_sensor_data(self):
        while True:
            try:
                self.ser = serial.Serial(self.config.port1, baudrate=115200, timeout=1)
                return self._receive_sensor_data(self.ser)
            except serial.serialutil.SerialException as e:
                print(f"Error: {e}")
                print(f"Waiting for {self.config.retry_delay} seconds before retrying...")
                for second in range(self.config.retry_delay, 0, -1):
                    print("{second}...")
                    time.sleep(1)
                    if second == 1:
                        break
            finally:
                if self.ser is not None:
                    self.ser.close()

    def _receive_sensor_data(self, ser):

        format_string = '!10i 2f 3f'  # 10 integers, 2 floats, 3 floats

        # Read the packed data from the serial port
        packed_data = ser.read(struct.calcsize(format_string))

        # Unpack the received data
        unpacked_data = struct.unpack(format_string, packed_data)

        # Convert the tuple to a list and return
        return list(unpacked_data)
