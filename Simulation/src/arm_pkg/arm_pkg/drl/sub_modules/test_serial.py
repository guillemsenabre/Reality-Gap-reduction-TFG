import serial
import time

class MoveJoints:
    def __init__(self, port1, port2):
        self.ser1 = serial.Serial(port1, baudrate=115200, timeout=1)
        self.ser2 = serial.Serial(port2, baudrate=115200, timeout=1)

    def send_values(self, values):
        if len(values) != 10:
            print("Error: You need to provide exactly 10 float values.")
            return

        values1 = values[:5]
        values2 = values[5:]

        try:
            self._send_list(self.ser1, values1)
            time.sleep(1)  # Ensure a small delay between sending to different ports
            self._send_list(self.ser2, values2)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.ser1.close()
            self.ser2.close()

    def _send_list(self, ser, values):
        values_str = ','.join(map(str, values))
        ser.write(values_str.encode())
        print(f"Sent values to {ser.port}: {values_str}")
