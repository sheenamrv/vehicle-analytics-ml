import socket
import json
import time

HOST = "127.0.0.1"
PORT = 5005
# Send one sample every 0.1 seconds
INTERVAL_SECONDS = 0.1

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send 100 sample packets
for i in range(100):
    # Use a relative timestamp so the stream starts at 0 seconds
    packet = {
        "timestamp": round(i * INTERVAL_SECONDS, 3),
        "s1": i,
        "s2": i + 1,
        "s3": i + 2,
        "s4": i + 3,
        "s5": i + 4,
        "s6": i + 5,
    }

    sock.sendto(json.dumps(packet).encode(), (HOST, PORT))
    print(f"Sent packet {i + 1}")
    time.sleep(INTERVAL_SECONDS)

sock.close()
print("Finished.")