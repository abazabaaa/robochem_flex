"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: testing function for the listener on server side

"""

import unittest
from unittest.mock import patch, MagicMock
import socket
import json
from threading import Thread
import time
import sys

# from server_app.RamaBerry_listener import RamaBerryListener


class TestRamaBerryListener(unittest.TestCase):
    host = "127.0.0.1"
    port = 65432
    timeout = 2

    @classmethod
    def setUpClass(cls):
        cls.avalanche_mock = MagicMock()
        sys.modules["server_app.RamaBerry_listener.AVAlanCHE"] = cls.avalanche_mock

        global RamaBerryListener
        from server_app.RamaBerry_listener import RamaBerryListener as rbl

    @classmethod
    def tearDownClass(cls):
        del sys.modules["server_app.RamaBerry_listener.AVAlanCHE"]

    def setUp(self):
        """setup the test case"""
        self.thread_running = True
        # setup the thread object to run the function ramaberry_listener
        self.ramaberry_thread = Thread(target=self.ramaberry_listener)
        # start the thread
        self.ramaberry_thread.start()
        time.sleep(2)
        # connect to the server
        self.sock = self.connection()

    def ramaberry_listener(self):
        # this is the actual thread, we start the "server" from the ramaberry_listens

        ramalisten = RamaBerryListener()
        ramalisten.server_start()

    def tearDown(self):
        """tear down the test case"""
        # send the shutdown command
        self.send_command("shutdown")
        self.sock.close()

        # wait for the thread to finish
        self.ramaberry_thread.join()

        # kills the thread

    def connection(self):
        """connect to the server"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        return sock

    # define a method that sends the commands:
    def send_command(self, command, data=None):
        """send a command to the server"""
        payload = {"command": command, "data": data}

        self.sock.sendall(json.dumps(payload).encode("utf-8"))
        response = self.sock.recv(1024).decode("utf-8")
        return response

    # actual tests:
    def test_blank_test(self):
        """just test that the server is opened and closed (kind of a test of the test)"""
        self.assertTrue(True)

    def test_setup_command(self):
        """test that the right response is sent back when the setup command is sent"""
        response = self.send_command("setup")
        print(response)
        self.assertEqual(
            response, json.dumps({"status": "success", "message": "setup complete"})
        )

    def test_spectrometer_setup(self):
        """tests the spectrometer setup command"""
        data = {"integration_time": 100, "averages": 10}
        response = self.send_command("spectrometer_setup", data)
        print(response)
        self.assertEqual(
            response,
            json.dumps(
                {"status": "success", "message": f"spectrometer setup with {data}"}
            ),
        )

    def test_fire_spectrum(self):
        """tests the fire spectrum command"""
        data = {"sample_name": "test_sample"}
        response = self.send_command("fire_spectrum", data)
        print(response)
        self.assertEqual(
            response,
            json.dumps(
                {
                    "status": "success",
                    "message": f'spectrum fired for {data["sample_name"]}',
                }
            ),
        )

    def test_make_model(self):
        """tests the make model command"""
        response = self.send_command("make_model")
        print(response)
        self.assertEqual(
            response, json.dumps({"status": "success", "message": f"model made"})
        )

    def test_quantify(self):
        """tests the quantify command"""
        response = self.send_command("quantify")
        print(response)
        self.assertEqual(
            response,
            json.dumps(
                {
                    "status": "success",
                    "message": f"quantification complete",
                    "yield": 0.99,
                }
            ),
        )

    def test_multiple_commands(self):
        """tests that if multiple commands are sent the server responds to all of them"""
        # setup
        response = self.send_command("setup")
        print(response)
        self.assertEqual(
            response, json.dumps({"status": "success", "message": "setup complete"})
        )
        time.sleep(2)
        # spectrometer setup
        data = {"integration_time": 100, "averages": 10}
        response = self.send_command("spectrometer_setup", data)
        print(response)
        self.assertEqual(
            response,
            json.dumps(
                {"status": "success", "message": f"spectrometer setup with {data}"}
            ),
        )

    def test_unknown_command(self):
        """tests that if an unknown command is sent the server responds with an error"""
        response = self.send_command("BS_command")
        print(response)
        self.assertEqual(
            response,
            json.dumps(
                {"status": "error", "message": f"command BS_command not recognised"}
            ),
        )
