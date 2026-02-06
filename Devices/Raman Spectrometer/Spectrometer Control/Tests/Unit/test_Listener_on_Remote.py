"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: This is a script to run from my laptop when i want to test the spectrometer server side.

"""

host = "1.1.1.1"
port = 1234
import socket
import json


def send_command(sock, command, data=None):
    """send a command to the server"""
    payload = {"command": command, "data": data}

    sock.sendall(json.dumps(payload).encode("utf-8"))
    response = sock.recv(1024).decode("utf-8")
    return response


params = {
    "integration_time": 1000.0,
    "n_averages": 1,
    "n_scans": 1,
    "save_each_n": 1,
    "correct_dark": True,
    "save_as": "single",
    "safety_threshold": 1000,
    "save_moniker": "Test",
}
if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        response = send_command(s, "setup")
        print(response)
        response = send_command(s, "spectrometer_setup", params)
        print(response)
        response = send_command(s, "acq_data")
        print(response)

    print("Received", repr(data))
