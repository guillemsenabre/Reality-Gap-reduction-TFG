import time
import serial
import struct
import random

class States():
    def __init__(self):
        print("Initializing States module fiiisss")
        self.ser = None
        self.retry_delay = 4
        time.sleep(0.1)


    def read_sensor_data(self, port, number_motors=8, number_sensors=13):
        while True:
            try:
                self.ser = serial.Serial(port, baudrate=115200, timeout=1)
                return self._receive_sensor_data(self.ser, number_motors, number_sensors)
            
            #except serial.serialutil.SerialException as e: # --> linux
            except Exception as e: # --> Windows (it's a general exception)
                print(f"Error: {e}")
                print(f"Waiting for {self.retry_delay} seconds before retrying...")
                for second in range(self.retry_delay, 0, -1):
                    print(f"{second}...")
                    time.sleep(1)
                    if second == 1:
                        break
            finally:
                if self.ser is not None:
                    self.ser.close()
                    
    def _receive_sensor_data(self, ser, number_motors, number_sensors):
        format_string = f'<{number_motors}i {number_sensors}f'
        expected_size = struct.calcsize(format_string)
        count = 0
        print("Waiting for state data...")
        while ser.in_waiting < expected_size:
            print(f"{count}...")
            time.sleep(1)
            count += 1

        # Read the packed data from the serial port
        packed_data = ser.read(expected_size)

        # Unpack the received data
        unpacked_data = struct.unpack(format_string, packed_data)

        # Convert the tuple to a list
        unpacked_list_data = list(unpacked_data)

        # Normalize and process the states
        processed_states = self._process_states(unpacked_list_data, number_motors, number_sensors)

        #return unpacked_list_data
        return processed_states


    def _process_states(self, states, nmotors, nsensors):

        # Check all data has been received
        if len(states) == (nmotors + nsensors):

            # Calculate min and max values
            min_value = min(states)
            max_value = max(states)

            # Normalize between 0 and 100
            norm_states = [((1+(val - min_value)) / (max_value - min_value)) * 100 for val in states]

            # To use tanh normalization, uncomment the following line and import Torch
            # norm_states = torch.tanh(norm_states)

            return norm_states
        
        else:
            print("Not all states have been received, adding average values")

            # Filling the states list with average values
            num_random_values = (nmotors + nsensors) - len(states)
            avg_states = sum(states)/len(states)
            random_values = [random.uniform(avg_states - 1.0, avg_states + 1.0) for _ in range(num_random_values)]
            twisted_states = states + random_values

            # Calculate min and max values
            min_value = min(twisted_states)
            max_value = max(twisted_states)

            # Normalize between 0 and 100
            twisted_norm_states = [((val - min_value) / (max_value - min_value)) * 100 for val in twisted_states]

            # To use tanh normalization, uncomment the following line and import Torch
            # twisted_norm_states = torch.tanh(norm_states)

            return twisted_norm_states
