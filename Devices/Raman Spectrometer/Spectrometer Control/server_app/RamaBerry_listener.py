"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:
This is the main script on server side. It listens to the commands from robochem and executes them when they are received.
it also sends back the responses to Robochem. The commands range from, take a spectrum, make models and quantify with RAMA-LAMA.

"""

import sys
import os

dir_path = os.getcwd()
sys.path.append(dir_path)

import socket
import json
import argparse
import logging
from datetime import datetime

from AVAlanCHE.AVAlanCHE_class import AVAlanCHE
import pandas as pd
import threading
import queue
import time
from utils import setup_logger


def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise e

    return wrapper


class RamaBerryListener:
    def __init__(self, host, port):
        self.log = setup_logger(self.__class__.__name__, "listener")
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.addr = None

        self.running = False
        self.acq_thread = None  # Initialize acquisition thread
        self.acq_thread_event = (
            threading.Event()
        )  # For auto releasing of threads# For auto releasing of threads
        self.killer_thread = None
        self.server_start()

    def _handle_command(self, data):
        """handles the command received from the client"""
        try:
            self.log.info(f"Received command data: {data}")
            command = data.get("command", "no_command")
            command_data = data.get("data", None)
            self.log.debug(f"Parsed command: {command}, Command data: {command_data}")

            match command:
                case "spectrometer_setup":
                    self.log.info("Executing 'spectrometer_setup' command.")
                    return self._spec_setup()
                case "set_params":
                    self.log.info("Executing 'set_params' command.")
                    return self._set_parameters(command_data)
                case "start_acq":
                    self.log.info("Executing 'start_acq' command.")
                    return self._acq_data()
                case "poll_data":
                    self.log.info("Executing 'poll_data' command.")
                    return self._poll_data()
                case "stop_acq":
                    self.log.info("Executing 'stop_acq' command.")
                    return self._stop_acq()
                case "shutdown":
                    self.log.info("Executing 'shutdown' command. Stopping server.")
                    return {"status": "success", "message": "Hasta La Vista, Baby!"}
                case _:
                    self.log.warning(f"Unrecognized command: {command}")
                    return {
                        "status": "error",
                        "message": f"Command '{command}' not recognised.",
                    }
        except Exception as e:
            self.log.exception(f"Error handling command: {data} - Exception: {e}")
            return {"status": "error", "message": f"Internal server error: {str(e)}"}

    def _handle_connection(self):
        """Handles a single connection from a client."""
        self.log.info("Started handling a client connection.")
        while True:
            try:
                data = self.conn.recv(1024)
                if not data:
                    self.log.info("No data received. Closing connection.")
                    break

                self.log.debug(f"Raw data received: {data}")
                try:
                    decoded_data = json.loads(data.decode("utf-8"))
                    self.log.info(f"Decoded data: {decoded_data}")
                    response = self._handle_command(decoded_data)
                    to_send = json.dumps(response)
                    self.conn.sendall(len(to_send).to_bytes(4, "big"))
                    self.conn.sendall(to_send.encode("utf-8"))
                    self.log.debug(f"Response sent: {response}")

                    if decoded_data.get("command") == "shutdown":
                        self.log.info("Shutdown command received. Stopping server.")
                        self.server_stop()
                        break
                except json.JSONDecodeError:
                    error_message = {
                        "status": "error",
                        "message": "Command not recognised",
                    }
                    self.log.warning(f"JSONDecodeError: Invalid JSON format received.")
                    self.conn.sendall(json.dumps(error_message).encode("utf-8"))
            except Exception as e:
                self.log.exception(f"Error while handling connection: {e}")
                break
        self.log.info("Connection handling finished.")

    def server_start(self):
        """Starts the server."""
        try:
            self.log.info(f"Starting server on {self.host}:{self.port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen()
            self.log.info(
                f"Server listening on {self.host}:{self.port}. Waiting for connections..."
            )

            self.conn, self.addr = self.sock.accept()
            self.log.info(f"Connection established with {self.addr}")
            with self.conn:
                self._handle_connection()
        except Exception as e:
            self.log.exception(f"Error starting server: {e}")
        finally:
            self.log.info("Shutting down the server.")
            self.server_stop()

    def server_stop(self):
        """Stops the server."""
        self.log.info("Initiating server shutdown...")
        try:
            if self.conn:
                self.conn.close()
                self.log.info("Client connection closed.")
            if self.sock:
                self.sock.close()
                self.log.info("Server socket closed.")
            if hasattr(self, "spec") and self.spec:
                self._spec_close()
                self.log.info("Spectrometer closed.")
        except Exception as e:
            self.log.exception(f"Error during server shutdown: {e}")
        finally:
            self.log.info("Server shutdown completed.")

    def _spec_setup(self):
        """Sets up the spectrometer."""
        self.log.info("Setting up spectrometer...")
        try:
            self.spec = AVAlanCHE()
            self.log.info("Spectrometer setup successfully.")
            return {"status": "success", "message": "Spectrometer setup completed."}
        except Exception as e:
            self.log.exception(f"Error in setting up the spectrometer: {e}")
            return {
                "status": "error",
                "message": f"Error in setting up the spectrometer: {e}",
            }

    def _spec_close(self):
        """Tears down the spectrometer."""
        self.log.info("Closing spectrometer...")
        try:
            self.spec.kill_process()
            self.log.info("Spectrometer closed successfully.")
            return {"status": "success", "message": "Spectrometer closed."}
        except Exception as e:
            self.log.exception(f"Error in closing the spectrometer: {e}")
            return {
                "status": "error",
                "message": f"Error in closing the spectrometer: {e}",
            }

    def _set_parameters(self, parameters):
        """Feeds the parameters to the spectrometer."""
        self.log.info("Setting parameters for the spectrometer.")
        try:
            self.spec.set_parameters(parameters)
            self.log.info(f"Parameters set successfully: {parameters}")
            return {"status": "success", "message": "Parameters set."}
        except Exception as e:
            self.log.exception(f"Error in setting parameters: {e}")
            return {
                "status": "error",
                "message": f"Error in setting parameters: {e}",
            }

    def _data_extract(self):
        """Extracts data from the spectrometer."""
        self.log.debug("Extracting data from the spectrometer.")
        try:
            data = self.spec.data_server

            if isinstance(data, str):
                self.log.info(f"Data extracted as string: {data}")
                return data
            elif isinstance(data, pd.DataFrame):
                self.log.info("Data extracted as DataFrame.")
                return data.to_json(orient="columns")
            elif data is None:
                self.log.warning("No data available from the spectrometer.")
                return "no data available"
            else:
                error_msg = "Data not in the right format."
                self.log.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            self.log.exception(f"Error in extracting data: {e}")
            return {"status": "error", "message": f"Error in extracting data: {e}"}

    def _poll_data(self):
        """Polls the spectrometer for data."""
        self.log.info("Polling data from the spectrometer.")
        try:
            data = self._data_extract()
            if data == "waiting":
                self.log.info("Spectrometer is waiting for poller.")
                return {"status": "success", "message": "waiting"}
            if data == "done":
                self.log.info("Measurement completed.")
                return {"status": "success", "message": "done"}
            if data is not None:
                self.log.info("Data acquired successfully.")
                return {
                    "status": "success",
                    "message": "data acquired",
                    "data": data,
                }
            else:
                self.log.warning("No data available; spectrometer not measuring.")
                return {
                    "status": "fail",
                    "message": "no data, spectrometer not measuring",
                }
        except Exception as e:
            self.log.exception(f"Error in polling data: {e}")
            return {"status": "error", "message": f"Error in polling data: {e}"}

    def _acq_data(self):
        """Acquires the data from the spectrometer."""
        self.log.info("Starting data acquisition.")
        self.running = True
        try:
            if self.acq_thread is None or not self.acq_thread.is_alive():
                self.log.debug("Starting acquisition thread.")
                self.acq_thread_event.clear()  # Ensure no early termination
                self.acq_thread = threading.Thread(
                    target=self._acq_process, name="AcqThread"
                )
                self.acq_thread.start()
            if self.killer_thread is None or not self.killer_thread.is_alive():
                self.log.debug("Starting killer thread.")
                self.killer_thread = threading.Thread(
                    target=self._killer_thread, daemon=True, name="KillerThread"
                )
                self.killer_thread.start()
            self.log.info("Data acquisition started successfully.")
            return {"status": "success", "message": "acquisition started"}
        except Exception as e:
            self.log.exception(f"Error in starting acquisition: {e}")
            self.running = False
            return {"status": "error", "message": f"Error in acquisition: {e}"}

    def _acq_process(self):
        """Auxiliary function for the acquisition process."""
        self.log.info("Acquisition process initiated.")
        try:
            self.spec.setup_measurement()
            self.log.debug("Measurement setup completed.")
            self.spec.get_dark()  # Get dark current data
            self.log.debug("Dark current data acquired.")
            self.spec.run_measurement()  # Start measurement
            self.log.info("Measurement running.")
            self.running = False  # Mark acquisition as completed
            self.log.info("Acquisition process completed.")
        except Exception as e:
            self.running = False
            self.log.exception(f"Error in acquisition process: {e}")
            raise e
        finally:
            self.log.debug("Setting acquisition thread event.")
            self.acq_thread_event.set()

    def _killer_thread(self):
        """Releases the threads from their waiting state."""
        self.log.debug("Killer thread is waiting for the acquisition to finish.")
        self.acq_thread_event.wait()
        if self.acq_thread is not None:
            self.log.debug("Joining acquisition thread.")
            self.acq_thread.join()
            self.acq_thread = None
        self.log.info("Killer thread completed its execution.")

    @error_handler
    def _stop_acq(self):
        """Stops the acquisition."""
        self.log.info("Stopping data acquisition.")
        try:
            self.spec.stop()
            self.log.debug("Spectrometer stopped.")
            if self.acq_thread is not None:
                self.log.debug("Joining acquisition thread.")
                self.acq_thread.join()
                self.acq_thread = None
            self.running = False
            self.log.info("Acquisition stopped successfully.")
            return {"status": "success", "message": "acquisition stopped"}
        except Exception as e:
            self.log.exception(f"Error in stopping acquisition: {e}")
            return {"status": "error", "message": f"Error in stopping acquisition: {e}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RamaBerry Server Application")
    parser.add_argument("--host", type=str, help="Host IP address", required=False)
    parser.add_argument("--port", type=int, help="Port number", required=False)
    parser.add_argument(
        "--interface",
        type=str,
        choices=["ethernet", "wifi"],
        default="ethernet",
        help="Network interface to use (default: ethernet)",
    )

    args = parser.parse_args()

    # automatic ip detection if no arguments:
    if not args.host:
        if args.interface == "wifi":
            interface = "wlan0"
        else:
            interface = "eth0"

        import netifaces as ni

        try:
            args.host = ni.ifaddresses(interface)[ni.AF_INET][0]["addr"]
        except KeyError:
            print(f"Interface {interface} not found")
            sys.exit(1)

    if not args.port:
        args.port = 65432
    print(f"Starting server on {args.host}:{args.port}")
    listener = RamaBerryListener(args.host, args.port)
