import socket

host = 'your-esp32-ip'  # Replace with the ESP32's IP address
port = 80

while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        data = s.recv(1024)
        print(data.decode('utf-8'))
