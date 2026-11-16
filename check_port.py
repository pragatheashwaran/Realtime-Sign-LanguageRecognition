import socket
import sys

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(2)
            s.connect(('127.0.0.1', port))
            print(f"Port {port} is OPEN")
            return True
        except:
            print(f"Port {port} is CLOSED")
            return False

if __name__ == "__main__":
    check_port(5000)
