import time
import serial
import struct

class MoveJoints:
    def __init__(self):
        print("Initializing Move Joints module...")

        # Pause execution for 0.3 seconds to allow initialization to complete
        time.sleep(0.3)

    def move_joints(self, actions, port, number_motors=8):
        """
        Move the specified joints based on the provided actions.

        :param actions: List of joint movement values.
        :param port: Serial port for communication.
        :param number_motors: Number of motors or joints; default is 10.
        """
        
        # Establish a serial connection with the specified port at a baud rate of 115200
        self.ser = serial.Serial(port, baudrate=115200, timeout=1)

        # Check if the number of actions is even; if not, raise an error and return
        if len(actions) % 2 != 0:
            print("Error: there has to be an even amount of values")
            return

        try:          
            # Send the list of joint values to the specified serial port 
            self._send_list(actions, number_motors)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Close the serial connection, regardless of success or failure
            self.ser.close()

    def _send_list(self, values, number_motors):
        """
        Send a list of values as a packed binary structure over the serial connection

        :param values: List of values to be sent
        :param number_motors: Number of motors
        """

        # Pack the list of values into a binary structure and send it over the serial connection
        self.ser.write(struct.pack(f'<{number_motors}f', *values))
        print(f"Sent values to {self.ser.name}: {values}")

    def reset_motors(self):
        """
        Reset the connected motors using a specific command
        """
        try:
            self.ser.write(b'motors init\n')
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.ser.close()
