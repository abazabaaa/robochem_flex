/**
 * Definitions for controlling a 0-10v analog output.
 *
 * Author: Simone Pilon <s.pilon at uva.nl> - Noël Research Group - 2024
 */

#include "Analog_out.h"
#include "PID.h"
#include <Arduino.h>
#include <math.h>

uint8_t AnalogOut::error = ANALOGOUT_ERROR_OK;

AnalogOut::AnalogOut(uint8_t pwm_out_pin, uint8_t analog_in_pin)
{
	calibration = DEFAULT_CALIBRATION;
	output_pin = pwm_out_pin;
	input_pin = analog_in_pin;
	active = false;
}

void AnalogOut::init()
{
	pinMode(input_pin, INPUT);
	pinMode(output_pin, OUTPUT);
	analogWrite(output_pin, 0);
}

uint8_t AnalogOut::get_setpoint()
{
	return static_cast<uint8_t>(lround(pid_control.get_setpoint()/calibration));
}

uint8_t AnalogOut::get_signal()
{
	return analogRead(input_pin) >> 2;
}

void AnalogOut::set_setpoint(uint8_t value)
{
	float setpoint = value * calibration;
	pid_control.set_setpoint(setpoint);
	active = value != 0;
	if (!active)
	{
		analogWrite(output_pin, 0);
	}
}

void AnalogOut::update()
{
	if (active)
	{
		float input = analogRead(input_pin) / 4.0;
		analogWrite(output_pin, pid_control.output(input));
	}
}

void AnalogOut::set_P(float p)
{
	pid_control.kp = p;
}

float AnalogOut::get_P()
{
	return pid_control.kp;
}

void AnalogOut::set_I(float i)
{
	pid_control.ki = i;
}

float AnalogOut::get_I()
{
	return pid_control.ki;
}

void AnalogOut::set_D(float d)
{
	pid_control.kd = d;
}

float AnalogOut::get_D()
{
	return pid_control.kd;
}
