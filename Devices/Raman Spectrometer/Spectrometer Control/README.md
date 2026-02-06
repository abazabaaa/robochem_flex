# <img src="gui_app/logoRBA.png" alt="RBA Logo" height="100" style="vertical-align: middle;"> RamaBerry 

> ***"I find your lack of signal disturbing.”***  
> — *Adapted from Darth Vader (Star Wars)*

# RamanBerry_server
RamaBerry server is an api that allows to 1) use the raman spectrometer without having to relay on the Avaspec software. 2) As Avaspec software is not programmable we cannot use it in an automated fashion, therefore we need this solution. 

## Requirements:
1) Avantes raman spectrometer
2) Raspberry pi which has a avaspec.dll installed (documentation for this is not included in this repo due to licensing considerations, but the Avantes team is more than helpful to provide you with the necessary files and instructions)
3) Python 3.11 or whatever fits on your raspberry pi

## General Code setup:
The app should work  in two forms the first is the standalone app, this is run on the rasberry directly with it's own screen and all. The other is the server one, what do we mean by server? well the idea that the you can connect to the raspberry with an ethernet cable and send requests to it. Both cases the lower level is from the AVAlanCHE engine, this is a special class which handles the communication with the spectrometer.

## Standalone APP:
The app design prioritizes simplicity, minimizing the chances of malfunctions. It's a tkinter-based app, initiated from the root directory using:
```
bash
python gui_app/RamaberryAPP.py
```
On the left side of the app, you'll find a display for the generated data. On the right, there's a selector for input parameters. Start by setting a save folder. By default, it points to the UVA remote folder in a new folder created daily. For better organization, consider creating a new folder for each run. To name your folder, append the desired name to the path, omitting the trailing slash. Next, choose the saving method:

- single: Saves each file separately. This method is slower but consumes less RAM.
- monitor: Does not save data, useful for setup phases.
- kinetic: Saves a series of data as a single CSV file. It's faster but not suited for lengthy acquisitions due to the Raspberry Pi's limitations.

Proceed to set the acquisition parameters:

- Integration time: Duration for which the spectrometer's shutter remains open.
- Number of averages: The count of times a single spectrum is captured and averaged.
- Repetition number: The frequency of spectrum captures (0 for a single spectrum, 1 for two, and so on).
- Save each: An indirect method to introduce a delay between spectra during kinetics. Spectra are retained after each specified count. Thus, delay = save each * integration time * number of averages. Although it's not highly precise, the timestamping of spectra down to the millisecond offers a reliable temporal reference.

The app, by default, adjusts for the dark spectrum before each measurement starts. You can deactivate this feature by unchecking the corresponding box.

The 'Start acquisition' and 'Stop acquisition' functions are quite intuitive. Note that stopping the acquisition might incur a slight delay, depending on the polling loop's state when the button was pressed. However, the app ensures that unsaved data is stored upon halting the acquisition.

## Server APP:

The server application, despite its simplicity, is fully functional and serves as the core communication hub for the system. It's built on a robust socket interface.

To initiate the server side (operating in headless mode), use the following command at the start of the session:
```bash
./raman_server_launcher.sh -i [interface] -h [host] -p [port]
```
The interface is the network interface (ethernet or wifi) the server will use to communicate. The host ip is the ip of the raspberry pi, and the port is the port the server will listen to. Both the host and port are optional and the server will try to find it's own ip and port if not provided.

The server is programmed to recognize and execute the following commands:

- spectrometer_setup: Initializes the spectrometer. This command should be executed first.
- set_params: Configures the parameters required for the measurement, similar to the setup process.
- start_acq: Initiates the measurement process. Data is automatically stored, although it's advisable to specify a save path within the parameters.
- poll_data: Retrieves data if it's available and hasn't been previously accessed.
- stop_acq: Halts the measurement process. It's optimal to use the poll data command immediately following this to ensure all data is collected.
- shutdown: Terminates the connection with the spectrometer and shuts down the server.

To run a command first generate the payload by:
```python
payload = {'command': 'setup', 'data': None } #or the parameters 
```
Then send it to the server by:
```python
sock.sendall(json.dumps(payload).encode("utf-8"))
response = self.sock.recv(1024).decode("utf-8")
```
of course you need to have a socket open to the server.


This configuration is intentionally designed to offload the majority of computational and memory-intensive tasks from the Raspberry Pi, ensuring a smooth and efficient operation.

