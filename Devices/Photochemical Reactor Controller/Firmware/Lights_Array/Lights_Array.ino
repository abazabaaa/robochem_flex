/*
 Arduino UNO light intensity controller.
 With some external components, generates 0-10v smooth analog signals which can control the light intensity for any light sources supporting this
 feature. One Arduino can control up to 4 light sources individually.
 The controller has been tested with PR160L light sources.

 Serial Communication:

   Sx=y
   Set variable x to value y. Variable number are integer, values are signed floating point.
   Rx
   Read variable x and print its value to serial.

 Variable list:
   number | acccess     | type        | description
   -------|-------------|-------------|--------------------------
    0     | RESERVED    |             | Serial.parseInt returns 0 on error.
    1     | READ/WRITE  | STRING [20] | Device identifier.
    2     | READ_ONLY   | INT[0-254]  | Number of subdevices.
    3     | READ/WRITE  | INT[0-1]    | Enable power source for lights (all lights).
    4     | READ_ONLY   | INT[0-5000] | Power source measured current (all lights) [mA].
    5     | READ/WRITE  | INT[0-1024] | Current measure offset.
    6     | READ/WRITE  | INT         | Current measure proportional conversion factor.
    7     | READ_ONLY   | FLOAT       | Power source measured voltage (all lights) [V].
    8     | READ/WRITE  | FLOAT       | Voltage measure proportional conversion factor.
    9     | READ_ONLY   | INT-INT     | Error register 'x-y' x = error type, y = error value. Is reset upon reading.
   10     | COMMAND     | NONE        | Save defaults.
   11     | COMMAND     | NONE        | Factory Reset.

 Array Variable list (repeats for each light source):
   Variable number = 20+10*subdevice_id+index
   index | acccess     | type        | description
   ------|-------------|-------------|-------------------------
    0    | READ/WRITE  | INT[0-100]  | Light intensity [%].
    1    | READ/WRITE  | FLOAT       | Calibration value.
    2    | READ/WRITE  | FLOAT       | PID Proportional constant.
    3    | READ/WRITE  | FLOAT       | PID Integral constant.
    4    | READ/WRITE  | FLOAT       | PID Derivative constant.

 Error codes:
   type  |  description    | value
   ------|-----------------|--------------
   0     | No error        | Undefined
   1     | Serial error    | Variable number which gave the issue
   2     | Subdevice error | See Analog_out.h

 Hardware connections (Arduino UNO board):
   Todo

 Libraries:
   no external libraries.

 by Simone Pilon <s.pilon at uva.nl> - Noël Research Group - 2024
*/

// todo ADD PID values to eeprom.
// todo add errors to devices (try to detect if 16v power is on).

#include <math.h>
#include <EEPROM.h>
#include "Device_id.h"
#include "Analog_out.h"

// Error codes (type)
#define ERROR_NO_ERROR 0	// No error.
#define ERROR_SERIAL 1     // error value is variable number from error.
#define ERROR_ANALOGOUT 2  // See Analog_out.h

// Other
#define MAX_SUBDEVICES 4
#define SERIAL_ARRAY_BEGIN 20  // First array command in serial
#define SERIAL_ARRAY_PERIOD 10 // Number of serial commands per sensor

// General pins
#define SSRELAY_PIN 4          // Activate solid state relay (lights power enable).
#define CURRENT_SENS_PIN 19    // Measure power current.
#define VOLTAGE_SENS_PIN 18    // Measure power voltage.

// Subdevices pins
// Do not use 9, 10 as pwm out (Timer 1 used for timing),
#define SUB0_OUTPUT_PIN 5
#define SUB0_INPUT_PIN 14	// AN0
#define SUB1_OUTPUT_PIN 6
#define SUB1_INPUT_PIN 15	// AN1
#define SUB2_OUTPUT_PIN 11
#define SUB2_INPUT_PIN 16	// AN2
#define SUB3_OUTPUT_PIN 3
#define SUB3_INPUT_PIN 17	// AN3

// EEPROM
#define EEPROM_ADDRESS_CHECK 0	// 1 byte char
#define DEFAULT_CHECK 42
#define EEPROM_ADDRESS_ID 1	// 20 bytes char[]
#define DEFAULT_ID "LIGHTS_0"
#define EEPROM_ADDRESS_CURRENT_OFFSET 30	// 2 byte int
#define DEFAULT_CURRENT_OFFSET 509
#define EEPROM_ADDRESS_CURRENT_CAL 32		// 2 byte int
#define DEFAULT_CURRENT_CAL 37
#define EEPROM_ADDRESS_VOLTAGE_CAL 40		// 4 byte float
#define DEFAULT_VOLTAGE_CAL 0.0185
// +-- Subdevices (repeats)
#define EEPROM_ADDRESS_SD_CAL 44	// 4 byte float
#define EEPROM_ADDRESS_SD_P 48	// 4 byte float
#define DEFAULT_SD_P 0.80
#define EEPROM_ADDRESS_SD_I 52	// 4 byte float
#define DEFAULT_SD_I 0.04
#define EEPROM_ADDRESS_SD_D 56	// 4 byte float
#define DEFAULT_SD_D 0.02
#define EEPROM_PERIOD_SD 16

// Others
#define TIMER_COUNTER 25000	// Set for 100mS interval updates

// Global variables
uint8_t error_type = 0;
uint8_t error_value = 0;
bool power_enable = false;
int current_offset = DEFAULT_CURRENT_OFFSET;
int current_cal = DEFAULT_CURRENT_CAL;
float voltage_cal = DEFAULT_VOLTAGE_CAL;
AnalogOut subdevices[MAX_SUBDEVICES] = {
									AnalogOut( SUB0_OUTPUT_PIN, SUB0_INPUT_PIN),
									AnalogOut( SUB1_OUTPUT_PIN, SUB1_INPUT_PIN),
									AnalogOut( SUB2_OUTPUT_PIN, SUB2_INPUT_PIN),
									AnalogOut( SUB3_OUTPUT_PIN, SUB3_INPUT_PIN) };

/*
 * Read in current sensor output voltage and convert it to mA.
 *
 * @return Current in mA. 
 */
int read_current()
{
	analogReference(DEFAULT);
	analogRead(CURRENT_SENS_PIN);
	int value = analogRead(CURRENT_SENS_PIN);
	value -= current_offset;	// Zero shift
	value *= current_cal;	// Convert to mA.
	analogReference(INTERNAL);
	analogRead(SUB0_INPUT_PIN);
	return value;
}

/*
 * Read in power voltage and convert to V.
 *
 * @return Power supply voltage in V. 
 */
float read_voltage()
{
	// Using the internal reference here.
	float value = analogRead(VOLTAGE_SENS_PIN);
	value *= voltage_cal;	// Convert to V.
	return value;
}

/**
 * Load stored values from EEPROM to system
 */
void load_defaults()
{
	uint8_t charvalue = 0;
	EEPROM.get(EEPROM_ADDRESS_CHECK, charvalue);
	if (charvalue == DEFAULT_CHECK)
	{
		eeprom_get_id(EEPROM_ADDRESS_ID);
		EEPROM.get(EEPROM_ADDRESS_CURRENT_OFFSET, current_offset);
		EEPROM.get(EEPROM_ADDRESS_CURRENT_CAL, current_cal);
		EEPROM.get(EEPROM_ADDRESS_VOLTAGE_CAL, voltage_cal);

		for (uint8_t i=0; i<MAX_SUBDEVICES; i++)
		{
			int offset = EEPROM_PERIOD_SD*i;

			float value = 0.0;
			EEPROM.get(EEPROM_ADDRESS_SD_CAL + offset, value);
			subdevices[i].calibration = value;

			EEPROM.get(EEPROM_ADDRESS_SD_P + offset, value);
			subdevices[i].set_P(value);
			EEPROM.get(EEPROM_ADDRESS_SD_I + offset, value);
			subdevices[i].set_I(value);
			EEPROM.get(EEPROM_ADDRESS_SD_D + offset, value);
			subdevices[i].set_D(value);
		}
	}
	else
	{
		factory_reset();
	}
}

/**
 * Store values from system to EEPROM as defaults
 */
void store_defaults()
{
	eeprom_put_id(EEPROM_ADDRESS_ID);
	EEPROM.put(EEPROM_ADDRESS_CURRENT_OFFSET, current_offset);
	EEPROM.put(EEPROM_ADDRESS_CURRENT_CAL, current_cal);
	EEPROM.put(EEPROM_ADDRESS_VOLTAGE_CAL, voltage_cal);

	for (uint8_t i=0; i<MAX_SUBDEVICES; i++)
	{
		int offset = EEPROM_PERIOD_SD*i;

		float value = subdevices[i].calibration;
		EEPROM.put(EEPROM_ADDRESS_SD_CAL + offset, value);

		value = subdevices[i].get_P();
		EEPROM.put(EEPROM_ADDRESS_SD_P + offset, value);
		value = subdevices[i].get_I();
		EEPROM.put(EEPROM_ADDRESS_SD_I + offset, value);
		value = subdevices[i].get_D();
		EEPROM.put(EEPROM_ADDRESS_SD_D + offset, value);
	}
}

/**
 * Reset stored values to original defaults
 */
void factory_reset()
{
	uint8_t charvalue = DEFAULT_CHECK;
	EEPROM.put(EEPROM_ADDRESS_CHECK, charvalue);

	init_id(DEFAULT_ID);
	current_offset = DEFAULT_CURRENT_OFFSET;
	current_cal = DEFAULT_CURRENT_CAL;
	voltage_cal = DEFAULT_VOLTAGE_CAL;
	for (uint8_t i=0; i<MAX_SUBDEVICES; i++)
	{
		subdevices[i].calibration = DEFAULT_CALIBRATION;
		subdevices[i].set_P(DEFAULT_SD_P);
		subdevices[i].set_I(DEFAULT_SD_I);
		subdevices[i].set_D(DEFAULT_SD_D);
	}
	
	store_defaults();
}

/**
 * Convert variable number to sensor_id and sensor_command.
 *
 * @param variable_number Takes the variable number as input and stores the sensor variable number as output.
 * @param sensor_id Outputs the number of the sensor.
 */
void parse_array_command(int &variable_number, int &sensor_id)
{
	variable_number -= SERIAL_ARRAY_BEGIN;
	sensor_id = variable_number / SERIAL_ARRAY_PERIOD;
	variable_number = variable_number % SERIAL_ARRAY_PERIOD;
}

/**
 * Checks serial line for commands and executes them.
 */
void parse_serial()
{
	while (Serial.available() > 0)
	{
		char command = Serial.read();
		int og_variable_number = 0;
		int variable_number = 0;
		int sub_id = 0;

		if (command == 'R')
		{
			// Read variable
			variable_number = Serial.parseInt();
			og_variable_number = variable_number;

			if (variable_number >= SERIAL_ARRAY_BEGIN)
			{
				// Read subdevice variable
				parse_array_command(variable_number, sub_id);
				if (sub_id < MAX_SUBDEVICES)
				{
					switch (variable_number)
					{
						case 0:
							// Read light intensity setvalue
							Serial.println(subdevices[sub_id].get_setpoint());
							break;
						case 1:
							// Read calibration value
							Serial.println(subdevices[sub_id].calibration);
							break;
						case 2:
							// Read PID Proportional constant
							Serial.println(subdevices[sub_id].get_P());
							break;
						case 3:
							// Read PID Integral constant
							Serial.println(subdevices[sub_id].get_I());
							break;
						case 4:
							// Read PID Derivative constant
							Serial.println(subdevices[sub_id].get_D());
							break;
						default:
							// Syntax error
							error_type = ERROR_SERIAL;
							error_value = og_variable_number;
							break;
					}
				}
				else
				{
					// Sensor index out of range
					error_type = ERROR_SERIAL;
					error_value = og_variable_number;
				}
			}
			else
			{
				// Read device-wide variable
				switch (variable_number)
				{
					case 1:
						// Device identifier
						Serial.println(device_id);
						break;
					case 2:
						// Number of subdevices
						Serial.println(MAX_SUBDEVICES);
						break;
					case 3 :
						// Power enable
						Serial.println(power_enable?1:0);	
						break;
					case 4:
						// Power current
						Serial.println(read_current());
						break;
					case 5:
						// Current measurement offset
						Serial.println(current_offset);
						break;
					case 6:
						// Current measurement conversion
						Serial.println(current_cal);
						break;
					case 7:
						// Power voltage
						Serial.println(read_voltage());
            break;
					case 8:
						// Voltage measurement conversion
						Serial.println(voltage_cal);
						break;
					case 9:
						// Error register
						Serial.print(error_type);
						Serial.print('-');
						Serial.println(error_value);
						error_type = ERROR_NO_ERROR;
						error_value = 0;
						break;
					default:
						// Syntax error
						error_type = ERROR_SERIAL;
						error_value = og_variable_number;
						break;
				}
			}
			if (AnalogOut::error != ANALOGOUT_ERROR_OK)
			{
				error_type = ERROR_ANALOGOUT;
				error_value = AnalogOut::error;
				AnalogOut::error = ANALOGOUT_ERROR_OK;
			}
		}
		else if (command == 'S')
		{
			// Write variable
			variable_number = Serial.parseInt();
			og_variable_number = variable_number;
			Serial.read();
			int variable_value_int = 0;
			float variable_value_float = 0;

			if (variable_number >= SERIAL_ARRAY_BEGIN)
			{
				// Write subdevice variable
				parse_array_command(variable_number, sub_id);
				if (sub_id < MAX_SUBDEVICES)
				{
					switch (variable_number)
					{
						case 0:
							// Write light intensity
							variable_value_int = Serial.parseInt();
							if (variable_value_int <= 100 && variable_value_int >= 0)
							{
								subdevices[sub_id].set_setpoint(variable_value_int);
							}
							else
							{
								error_type = ERROR_SERIAL;
								error_value = og_variable_number;
							}
							break;
						case 1:
							// Write calibration value
							variable_value_float = Serial.parseFloat();
							subdevices[sub_id].calibration = variable_value_float;
							break;
						case 2:
							// Write PID Proportional constant
							variable_value_float = Serial.parseFloat();
							subdevices[sub_id].set_P(variable_value_float);
							break;
						case 3:
							// Write PID Integral constant
							variable_value_float = Serial.parseFloat();
							subdevices[sub_id].set_I(variable_value_float);
							break;
						case 4:
							// Write PID Derivative constant
							variable_value_float = Serial.parseFloat();
							subdevices[sub_id].set_D(variable_value_float);
							break;
						default:
							// Syntax error
							error_type = ERROR_SERIAL;
							error_value = og_variable_number;
							break;
					}
				}
				else
				{
					// Sensor index out of range
					error_type = ERROR_SERIAL;
					error_value = og_variable_number;
				}
			}
			else
			{
				// Write device-wide variable
				switch (variable_number)
				{
					case 1:
						// Device identifier
						serial_read_id();
						break;
					case 3:
						// Enable power
						variable_value_int = Serial.parseInt();
						power_enable = variable_value_int == 1;
						digitalWrite(SSRELAY_PIN, power_enable?HIGH:LOW);
						break;
					case 5:
						// Current measurement offset
						current_offset = Serial.parseInt();
						break;
					case 6:
						// Current measurement conversion
						current_cal = Serial.parseInt();
						break;
					case 8:
						// Voltage measurement conversion
						voltage_cal = Serial.parseFloat();
						break;
					case 10:
						// Save defaults
						store_defaults();
						store_defaults();
						break;
					case 11:
						// Factory reset
						factory_reset();
						break;
					default:
						// Syntax error
						error_type = ERROR_SERIAL;
						error_value = og_variable_number;
						break;
				}
			}
			if (AnalogOut::error != ANALOGOUT_ERROR_OK)
			{
				error_type = ERROR_ANALOGOUT;
				error_value = AnalogOut::error;
				AnalogOut::error = ANALOGOUT_ERROR_OK;
			}
		}
	}
}

void setup()
{
	// Load stored values from EEPROM
	load_defaults();

	// Initialize general pins
	pinMode(SSRELAY_PIN, OUTPUT);
	digitalWrite(SSRELAY_PIN, power_enable?HIGH:LOW);
	pinMode(CURRENT_SENS_PIN, INPUT);
	
	// Initialize sensors
	analogReference(INTERNAL);
	analogRead(SUB0_INPUT_PIN);	// FIrst value read after changing reference is wrong.
	for (uint8_t i=0; i<MAX_SUBDEVICES; i++)
	{
		subdevices[i].init();
	}

	// Setup timer 1
	// PWM 9 and 10 cannot be used.
	TCCR1A = 0;            // Init Timer1A
	TCCR1B = 0;            // Init Timer1B
	TCCR1B |= B00000011;   // Prescaler = 1:64
	OCR1A = TIMER_COUNTER; // Timer Compare1A Register (COMPA)
	TIMSK1 |= B00000010;   // Enable Timer COMPA Interrupt

	// Initialize Serial interface
	Serial.begin(9600);
}

void loop()
{
	while (true)
	{
		parse_serial();
	}
}

/**
 * Interrupt service routine for TIMER1 COMPA event.
 * Set to run every 100mS, updates PIDs.
 */
ISR(TIMER1_COMPA_vect)
{
	OCR1A += TIMER_COUNTER;
	for (uint8_t i=0; i<MAX_SUBDEVICES; i++)
	{
		subdevices[i].update();
	}
}
