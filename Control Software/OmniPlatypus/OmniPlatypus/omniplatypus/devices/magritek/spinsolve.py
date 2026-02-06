"""
File: spinsolve.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: Device communicating with Spinsolve localserver.
"""

import xml.etree.ElementTree as ET
from typing import Any, Union
import time

from omniplatypus.devices.base.device import (
    DeviceParameter,
    ParameterAccess,
    ParameterValueRun,
)
from omniplatypus.devices.base.device_socket import SocketDevice


class ElementXml:
    """
    Represent a simple XML element for the spinsolve requests.
    """

    name: str
    attributes: dict[str, str] | None
    content: list["ElementXml"] | str | None
    self_closing: bool

    def __init__(
        self,
        name: str,
        content: Union["ElementXml", list["ElementXml"], str, None] = None,
        attributes: dict[str, str] | None = None,
        self_closing: bool = False,
    ):
        self.name = name
        self.attributes = attributes
        if isinstance(content, ElementXml):
            content = [content]
        self.content = content
        self.self_closing = self_closing

    def to_string(self, indent: int = 0) -> str:
        attributes = ""
        indent_str = "  " * indent
        if self.attributes is not None:
            for key, value in self.attributes.items():
                attributes += f" {key}='{value}'"
        message = ""
        if self.content is not None and not self.self_closing:
            if isinstance(self.content, str):
                message = f"{indent_str}<{self.name}{attributes}>{self.content}</{self.name}>\n"
            else:
                message = f"{indent_str}<{self.name}{attributes}>\n"
                for element in self.content:
                    message += element.to_string(indent + 1)
                message += f"{indent_str}</{self.name}>\n"
        else:
            message = f"{indent_str}<{self.name}{attributes} />\n"
        return message


class SpinsolveProtocol:
    """
    Spinsolve protocol data container.
    """

    name: str
    options: dict[str, str]
    process: bool
    timeout: float

    def __init__(
        self,
        name: str = "1D EXTENDED+",
        options: dict[str, str] | None = None,
        process: bool = False,
        timeout: float = 60 * 10,
    ):
        """
        Constructor.
        If unsure about the options, use one of the factory methods to create an object.

        @param name: str = "1D EXTENDED+"
            Name of the protocol to use for the acquisition.
        @param options: dict | None = None
            Additional options for the measurement.
        @param process: bool = False,
            Include request to process result at the server side.
        @param timeout: float = 60*60
            Max time to wait before analysis is done.
        """
        self.name = name
        if options is None:
            options = {}
        self.options = options
        self.process = process
        self.timeout = timeout

    def __str__(self) -> str:
        message = f"{self.name} [process remotely: {self.process}]"
        if self.options is not None:
            message += "\n"
            for key, value in self.options.items():
                message += f"{key}: {value}\n"
        return message

    @classmethod
    def nmr_extended(cls) -> "SpinsolveProtocol":
        """
        Factory constructor for 1D nmr acquisition.
        """
        return SpinsolveProtocol(
            name="1D EXTENDED+",
            options={
                "PulseAngle": "90",
                "Number": "16",
                "RepetitionTime": "10",
                "AcquisitionTime": "1.6",
            },
            process=True,
            timeout=240.0,
        )

    @classmethod
    def nmr_19F(cls) -> "SpinsolveProtocol":
        """
        Factory constructor for 19F nmr acquisition.
        """
        # todo never tested
        # nevver checked options!
        return SpinsolveProtocol(
            name="1D FLUORINE",
            options={
                "PulseAngle": "90",
                "Number": "16",
                "RepetitionTime": "10",
                "AcquisitionTime": "1.6",
            },
            process=True,
            timeout=240.0,
        )

    @classmethod
    def nmr_19F_Hdec(cls) -> "SpinsolveProtocol":
        """
        Factory constructor for 19F nmr acquisition with broadband H decoupling.
        """
        return SpinsolveProtocol(
            name="1D FLUORINE HDEC",
            options={
                "Number": "32",
                "AcquisitionTime": "1.64",
                "RepetitionTime": "15",
                "PulseAngle": "90",
                "centerFrequency": "-180",
                "PulselengthScale": "1",
                "decouplePower": "0",
            },
            process=True,
            timeout=60.0 * 15,
        )

    @classmethod
    def nmr_1H(cls) -> "SpinsolveProtocol":
        """
        1H measurement protocol.
        """
        # todo never tested
        return SpinsolveProtocol(
            name="1D PROTON",
            options={
                "Scan": "QuickScan",  # options: QuickScan, StandardScan, PowerScan
            },
            process=True,
            timeout=240.0,
        )

    @classmethod
    def shim_check(cls) -> "SpinsolveProtocol":
        """
        Shim check protocol.
        """
        return SpinsolveProtocol(
            name="SHIM",
            options={"Shim": "CheckShim"},
            process=False,
            timeout=60.0,
        )

    @classmethod
    def shim_quick(cls) -> "SpinsolveProtocol":
        """
        Quick shim protocol.
        """
        return SpinsolveProtocol(
            name="SHIM",
            options={"Shim": "QuickShim"},
            process=False,
            timeout=60.0 * 6.0,
        )

    @classmethod
    def shim_power(cls) -> "SpinsolveProtocol":
        """
        Power-shim protocol.
        """
        return SpinsolveProtocol(
            name="SHIM",
            options={"Shim": "PowerShim"},
            process=False,
            timeout=60.0 * 60.0,
        )


class SpinsolveXmlParameter(DeviceParameter):
    """
    DeviceParameter with specific data for Spinsolve xml requests.
    """

    xml_header = "<?xml version='1.0' encoding='UTF-8'?>\n"
    end_of_response = "</Message>"

    value_in: ElementXml | None
    internal_id: ElementXml
    responses: int  # Number of 'Message' tags to expect

    def __init__(
        self,
        name: str,
        access_level: ParameterAccess,
        value_type: type,
        internal_id: ElementXml,
        value_goes_in: ElementXml | None,
    ) -> None:
        DeviceParameter.__init__(self, name, access_level, value_type, internal_id)
        self.value_in = value_goes_in
        self.responses = 0
        if access_level == ParameterAccess.R:
            self.responses = 1

    def xml_request(self, value: Any = None) -> str:
        if self.value_in is not None:
            if isinstance(value, SpinsolveProtocol):
                self.value_in.attributes = {"protocol": value.name}
                content = []
                if value.options is not None:
                    for option_name, option_value in value.options.items():
                        content.append(
                            ElementXml(
                                name="Option",
                                attributes={"name": option_name, "value": option_value},
                                self_closing=True,
                            )
                        )
                if value.process:
                    content.append(
                        ElementXml(
                            name="Processing",
                            content=ElementXml(
                                name="Press",
                                attributes={"Name": "MNOVA"},
                                self_closing=True,
                            ),
                        )
                    )
                if len(content) > 0:
                    self.value_in.content = content
            else:
                self.value_in.content = value
        return self.xml_header + self.internal_id.to_string()


class SpinsolveDataParameter(SpinsolveXmlParameter):
    """
    Parameter for 'Data' element-type parameters.
    """

    value_key: str
    value_in: ElementXml
    internal_id: ElementXml

    def __init__(
        self,
        name: str,
        access_level: ParameterAccess,
        value_type: type,
        value_key: str,
    ) -> None:
        value_in = ElementXml(
            name="Data",
            attributes={"key": value_key, "value": ""},
            self_closing=True,
        )
        internal_id = ElementXml(
            name="Message",
            content=ElementXml(
                name="Set",
                content=ElementXml(
                    name="UserData",
                    content=value_in,
                ),
            ),
        )
        SpinsolveXmlParameter.__init__(
            self,
            name=name,
            access_level=access_level,
            value_type=value_type,
            internal_id=internal_id,
            value_goes_in=value_in,
        )
        self.value_key = value_key

    def xml_request(self, value: Any = None) -> str:
        self.value_in.attributes["value"] = value
        return self.xml_header + self.internal_id.to_string()


class SpinsolveError(Exception):
    pass


class SpinsolveClient(SocketDevice):
    """
    Connect to Spinsolve server via socket interface and access NMR device.
    """

    filename_strf = "%Y-%m-%d_%H%M%S"
    # data on most used protocols
    protocols = {
        "1D EXTENDED+": {
            "Number": {
                "default": "16",
                "units": "",
                "values": [
                    "0",
                    "1",
                    "2",
                    "4",
                    "8",
                    "16",
                    "32",
                    "64",
                    "128",
                    "256",
                    "512",
                    "1024",
                    "2048",
                    "4096",
                    "8192",
                    "16384",
                ],
            },
            "AcquisitionTime": {
                "default": "6.4",
                "units": "S",
                "values": [
                    "0.4",
                    "0.8",
                    "1.6",
                    "3.2",
                    "6.4",
                ],
            },
            "RepetitionTime": {
                "default": "10",
                "units": "S",
                "values": [
                    "1",
                    "2",
                    "4",
                    "7",
                    "10",
                    "15",
                    "30",
                    "60",
                    "120",
                    "180",
                    "300",
                    "420",
                    "600",
                ],
            },
            "PulseAngle": {
                "default": "90",
                "units": "°",
                "values": ["30", "45", "60", "90"],
            },
        },
        "1D FLUORINE HDEC": {
            "Number": {
                "default": "128",
                "units": "",
                "values": [
                    "0",
                    "1",
                    "2",
                    "4",
                    "8",
                    "16",
                    "32",
                    "64",
                    "128",
                    "256",
                    "512",
                    "1024",
                    "2048",
                    "4096",
                    "8192",
                    "16384",
                    "32768",
                    "65536",
                    "131072",
                ],
            },
            "AcquisitionTime": {
                "default": "1.64",
                "units": "S",
                "values": ["0.102", "0.205", "0.41", "0.819", "1.64"],
            },
            "RepetitionTime": {
                "default": "10",
                "units": "S",
                "values": [
                    "0.3",
                    "0.5",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "15",
                    "30",
                    "60",
                    "120",
                    "180",
                    "240",
                    "300",
                    "360",
                    "420",
                    "480",
                    "540",
                    "600",
                ],
            },
            "PulseAngle": {
                "default": "90",
                "units": "°",
                "values": ["5", "10", "30", "45", "70", "90"],
            },
            "centerFrequency": {
                "default": -170,
                "units": "ppm",
            },
            "PulselengthScale": {
                "default": "1",
                "units": "",
                "values": ["1", "2", "4"],
            },
            "decouplePower": {
                "default": "0",
                "units": "dB",
                "values": [
                    "0",
                    "-1",
                    "-2",
                    "-3",
                    "-4",
                    "-5",
                    "-6",
                    "-7",
                    "-8",
                    "-9",
                    "-10",
                    "-11",
                    "-12",
                    "-13",
                    "-14",
                    "-15",
                    "-85",
                ],
            },
        },
    }
    all_protocols = [
        "NO PROTOCOL SELECTED",
        "Interface",
        "HISTORY",
        "STANDBY",
        "SHIM 1H SAMPLE",
        "SHIM RM",
        "SHIM",
        "MONITOR",
        "SCRIPT",
        "AUDIT LOG",
        "qNMR HISTORY",
        "SETUP",
        "SERVICE",
        "QUEUE",
        "qNMR",
        "FLUORINE T2",
        "FLUORINE T1",
        "1D PROTON FDEC",
        "PROTON-FLUORINE COSY",
        "1D FLUORINE HDEC",
        "1D FLUORINE",
        "1D FLUORINE+",
        "FLUORINE JRES 2D",
        "FLUORINE COSY 2D",
        "CARBON T1 IR",
        "HSQC WALTZ",
        "HSQC-ME-WALTZ",
        "HMQC WALTZ",
        "HMBC WALTZ",
        "1D CARBON HFDEC",
        "HETCOR WALTZ",
        "DEPT WALTZ",
        "1D CARBON WALTZ",
        "1D CARBON+ WALTZ",
        "APT WALTZ",
        "TOCSY",
        "T2",
        "T1",
        "ROESY",
        "RM",
        "1D PROTON CDEC",
        "1D PROTON",
        "1D PRESAT",
        "1D EXTENDED+",
        "1D PRESAT MULTI",
        "PARAMAGNETIC",
        "JRES 2D",
        "COSY 2D",
        "COSY+",
    ]

    # tags for status messages (sent by spinsolve server)
    _status_tag = "StatusNotification"
    _error_tag = "Error"
    _progress_tag = "Progress"
    _complete_tag = "Completed"
    _state_tag = "State"

    def __init__(self):
        SocketDevice.__init__(self)

        self.generic_name = "Spinsolve NMR Client"

        xml_value = ElementXml(name="Sample", content="")
        self.parameter_sample = SpinsolveXmlParameter(
            name="sample",
            access_level=ParameterAccess.W,
            value_type=str,
            internal_id=ElementXml(
                name="Message",
                content=ElementXml(name="Set", content=xml_value),
            ),
            value_goes_in=xml_value,
        )
        self.parameter_sample.description = (
            "Set the sample name for the next measurement."
        )
        self.add_parameter(self.parameter_sample)

        xml_value = ElementXml(name="Solvent", content="")
        self.parameter_solvent = SpinsolveXmlParameter(
            name="solvent",
            access_level=ParameterAccess.W,
            value_type=str,
            internal_id=ElementXml(
                name="Message",
                content=ElementXml(name="Set", content=xml_value),
            ),
            value_goes_in=xml_value,
        )
        self.parameter_solvent.description = "Set the solvent for the next measurement."
        self.add_parameter(self.parameter_solvent)

        self.parameter_comment = SpinsolveDataParameter(
            name="comment",
            access_level=ParameterAccess.W,
            value_type=str,
            value_key="Comment",
        )
        self.parameter_comment.description = "Set the comment for the next measurement."
        self.add_parameter(self.parameter_comment)

        xml_value = ElementXml(name="UserFolder", content="")
        self.parameter_folder = SpinsolveXmlParameter(
            name="data_folder",
            access_level=ParameterAccess.W,
            value_type=str,
            internal_id=ElementXml(
                name="Message",
                content=ElementXml(
                    name="Set",
                    content=ElementXml(
                        name="DataFolder",
                        content=xml_value,
                    ),
                ),
            ),
            value_goes_in=xml_value,
        )
        self.parameter_folder.description = (
            "Set the folder where the next measurement will be stored."
        )
        self.add_parameter(self.parameter_folder)

        xml_value = ElementXml(name="Start")
        self.parameter_start = SpinsolveXmlParameter(
            name="start",
            access_level=ParameterAccess.W,
            value_type=SpinsolveProtocol,
            internal_id=ElementXml(
                name="Message",
                content=xml_value,
            ),
            value_goes_in=xml_value,
        )
        self.parameter_start.responses = 105
        self.parameter_start.description = (
            "Start an NMR protocol, such as measure or shim."
        )
        self.add_parameter(self.parameter_start)

        xml_value = ElementXml(name="Script")
        self.parameter_script = SpinsolveXmlParameter(
            name="run_script",
            access_level=ParameterAccess.W,
            value_type=str,
            internal_id=ElementXml(
                name="Message",
                content=ElementXml(name="Execute", content=xml_value),
            ),
            value_goes_in=xml_value,
        )
        self.parameter_script.responses = 5
        self.parameter_script.description = "Execute a spinsolve script."
        self.add_parameter(self.parameter_script)

        self.parameter_abort = SpinsolveXmlParameter(
            name="abort",
            access_level=ParameterAccess.W,
            value_type=ParameterValueRun,
            internal_id=ElementXml(
                name="Message", content=ElementXml(name="Abort", self_closing=True)
            ),
            value_goes_in=None,
            # "<Message><Abort /></Message>",
        )
        self.parameter_abort.description = (
            "Abort an NMR protocol, such as measure or shim."
        )
        self.add_parameter(self.parameter_abort)

        self.parameter_protocols = SpinsolveXmlParameter(
            name="available_protocols",
            access_level=ParameterAccess.R,
            value_type=str,  # TODO proper for conversion
            internal_id=ElementXml(
                name="Message",
                content=ElementXml(name="AvailableProtocolsRequest", self_closing=True),
            ),
            value_goes_in=None,
            # "<Message>\n<AvailableProtocolsRequest/>\n</Message>\n"
        )
        self.parameter_protocols.description = "Request a list of supported protocols."
        self.add_parameter(self.parameter_protocols)

        self.parameter_options = []
        generate_for = {
            "extended": "1D EXTENDED+",
            "shim": "SHIM",
            "fluorine_hdec": "1D FLUORINE HDEC",
            "fluorine": "1D FLUORINE+",
        }
        for name, protocol_name in generate_for.items():
            option = SpinsolveXmlParameter(
                name="available_options_" + name,
                access_level=ParameterAccess.R,
                value_type=str,  # TODO proper for conversion
                internal_id=ElementXml(
                    name="Message",
                    content=ElementXml(
                        name="AvailableOptionsRequest",
                        attributes={"protocol": protocol_name},
                        self_closing=True,
                    ),
                ),
                value_goes_in=None,
                # "<Message>\n<AvailableOptionsRequest protocol='1D EXTENDED+'/>\n</Message>\n",
            )
            option.description = "Request a list of supported options for protocol xyz."
            self.parameter_options.append(option)
            self.add_parameter(option)

    @property
    def keep_open(self) -> bool:
        """
        Expresses a preference for opening the device for each operation, rather that opening at platform build-time
        and keeping it open (default).

        @return: bool
            True if the device should be kept in the open state when not in use.
        """
        return False

    # def is_open(self) -> bool:
    #     # todo debuugging
    #     return True

    #      def _to_xml(self, request: AcquisitionRequest) -> str:
    #          """
    #          Converts the acquisition request objects into xml strings.
    #
    #          @param request: AcquisitionRequest
    #              The object describing the request.
    #          @return: str
    #              Xml string to send to the Spinsolve server.
    #          """
    #          xml_request = self._xml_header
    #          xml_request += self._xml_message_set("Sample", request.sample_name)
    #          xml_request += self._xml_message_set("Solvent", request.solvent)
    #          comment = request.comment.strip("'")
    #          xml_request += self._xml_message_set(
    #              "UserData", f"<Data key='Comment' value='{comment}'/>"
    #          )
    #          xml_request += self._xml_message_set(
    #              ["DataFolder", "UserFolder"], request.file_path
    #          )
    #          xml_request += "<Message>\n"
    #          xml_request += f"    <Start protocol='{request.protocol}'>\n"
    #          if request.options is not None:
    #              for name, value in request.options.items():
    #                  xml_request += f"        <Option name='{name}' value='{value}'/>\n"
    #          if request.process:
    #              xml_request += "        <Processing>\n"
    #              xml_request += "            <Press Name='MNOVA'/>\n"
    #              xml_request += "        </Processing>\n"
    #          xml_request += "    </Start>\n"
    #          xml_request += "</Message>\n"
    #          return xml_request

    def _listen(
        self,
        parameter: SpinsolveXmlParameter,
        timeout: float,
        log: bool = False,
        protocol: SpinsolveProtocol | None = None,
    ):
        max_time = time.time() + timeout
        responses = parameter.responses
        message = ""
        while time.time() <= max_time and responses > 0:
            time.sleep(0.5)
            buffer = self._socket.recv(4096).decode("utf-8")
            if not buffer == "":
                max_time = time.time() + timeout
                if log:
                    self.log(f"Received:\n{buffer}")
                message += buffer
                buffer_parts = buffer.count(parameter.end_of_response)
                responses -= buffer_parts
                if buffer_parts > 1:
                    parts = buffer.split(parameter.end_of_response)
                    parts = [x + parameter.end_of_response for x in parts]
                else:
                    parts = [buffer]
                for part in parts:
                    try:
                        response_root = ET.fromstring(part)
                        update = response_root.find(self._status_tag)
                        if update is not None:
                            update_content = update[0]
                            if update_content.tag == self._error_tag:
                                description = update_content.attrib.get(
                                    "error", "no description found."
                                )
                                error = SpinsolveError(f"NMR Error: {description}")
                                self.log(error)
                                raise error
                            elif update_content.tag == self._progress_tag:
                                if (
                                    protocol is None
                                    or not update_content.attrib["protocol"]
                                    == protocol.name
                                ):
                                    error = SpinsolveError(
                                        f"NMR seems to be recording another spectrum ({update_content.attrib['protocol']})."
                                    )
                                    self.log(error)
                                    raise error
                                self.log(
                                    f"Progress: {update_content.attrib['percentage']}% ({update_content.attrib['secondsRemaining']} S left)"
                                )
                            elif update_content.tag == self._complete_tag:
                                if (
                                    protocol is None
                                    or not update_content.attrib["protocol"]
                                    == protocol.name
                                ):
                                    error = SpinsolveError(
                                        f"NMR seems to be recording another spectrum ({update_content.attrib['protocol']})."
                                    )
                                    self.log(error)
                                    raise error
                                if not (
                                    update_content.attrib["completed"] == "true"
                                    and update_content.attrib["successful"] == "true"
                                ):
                                    error = SpinsolveError(
                                        f"NMR measurement completed with an error."
                                    )
                                    self.log(error)
                                    raise error
                                self.log("Progress: completed.", level="ok")
                            elif update_content.tag == self._state_tag:
                                if (
                                    protocol is None
                                    or not update_content.attrib["protocol"]
                                    == protocol.name
                                ):
                                    error = SpinsolveError(
                                        f"NMR seems to be recording another spectrum ({update_content.attrib['protocol']})."
                                    )
                                    self.log(error)
                                    raise error
                                status = update_content.attrib["status"]
                                self.log(f"Status: {status}.")
                                if status == "Ready":
                                    # todo protocol is done
                                    responses = 0
                    except ET.ParseError as error:
                        self.log(error)
                    except IndexError:
                        self.log("Received empty status update.", level="warning")
        return message

    def _write(self, parameter: DeviceParameter, value: Any) -> None:
        """
        Low-level function to write data to the device.

        @param parameter: DeviceParameter
            The parameter which will be accessed.
        @param value:
            The value of the parameter to be written to the device.

        Note: needs to be overridden in subclass!
        """
        if isinstance(parameter, SpinsolveXmlParameter):
            # Send xml request
            request = parameter.xml_request(value)
            self._socket.sendall(request.encode())

            # Wait for response
            timeout = 10.0  # [S]
            protocol = None
            if isinstance(value, SpinsolveProtocol):
                timeout = value.timeout
                protocol = value
            self._listen(parameter, timeout, protocol=protocol)
        else:
            raise NotImplementedError(
                f"Cannot write '{type(parameter)}'-type parameter."
            )

    def _read(self, parameter: DeviceParameter) -> Any:
        """
        Low-level function to read data from the device.

        @param parameter: DeviceParameter
            The parameter which will be accessed.
        @return:
            The value of the parameter read from the device.

        Note: needs to be overridden in subclass!
        """
        if isinstance(parameter, SpinsolveXmlParameter):
            # Send xml request
            request = parameter.xml_request()
            self._socket.sendall(request.encode())

            # Wait for response
            return self._listen(parameter, 10.0)
        else:
            raise NotImplementedError(
                f"Cannot read '{type(parameter)}'-type parameter."
            )


if __name__ == "__main__":
    sp = SpinsolveClient()
    sp.initialize()
    sp["sample"] = "my sample"
    sp["solvent"] = 666
    sp["comment"] = "fggfr rtrt/"
    sp["start"] = SpinsolveProtocol.nmr_extended()
    # print(sp.parameter_protocols.internal_id.to_string())
    # print(sp.parameter_abort.internal_id.to_string())
