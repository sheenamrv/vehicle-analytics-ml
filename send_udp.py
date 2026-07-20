import socket
import json
import time

HOST = "127.0.0.1"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for i in range(100):
    packet = {
        "timestamp": time.time(),
        "s1": i,
        "s2": i+1,
        "s3": i+2,
        "s4": i+3,
        "s5": i+4,
        "s6": i+5,
    }

    sock.sendto(json.dumps(packet).encode(), (HOST, PORT))
    print(f"Sent packet {i + 1}")
    time.sleep(0.1)

sock.close()
print("Finished.")