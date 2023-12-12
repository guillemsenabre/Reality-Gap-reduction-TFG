import serial
import rclpy


# Open the serial connection (adjust port and baudrate based on your ESP32 configuration)
ser = serial.Serial('COM0', 115200)

try:
    while True:
        # Read a line from the serial port
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
except KeyboardInterrupt:
    pass
finally:
    ser.close()
