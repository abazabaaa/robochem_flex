/**
 * Simple PID controller implementation.
 *
 * Author: Simone Pilon <s.pilon at uva.nl> - Noël Research Group - 2024
 */

#include "PID.h"
#include <Arduino.h>

#define INTEGRAL_MAX 25500
#define INTEGRAL_MIN -25500

PID::PID()
{
	kp = 0.8;
	ki = 0.1;
	kd = 0.1;

	max_out = 255.0;
	min_out = 0.0;

	setpoint = 0.0;
	last_input = 0.0;
	sum_out = 0.0;
}

void PID::set_setpoint(float _setpoint)
{
	setpoint = _setpoint;

	//sum_out = 0;
}

float PID::get_setpoint()
{
	return setpoint;
}

float PID::output(float input)
{
	float error = setpoint - input;
	float delta_error = error - (setpoint - last_input);
	sum_out += error * ki;
	if (sum_out > INTEGRAL_MAX)
	{
		sum_out = INTEGRAL_MAX;
	}
	else if (sum_out < INTEGRAL_MIN)
	{
		sum_out = INTEGRAL_MIN;
	}
	last_input = input;
	float output = error*kp + sum_out + delta_error*kd;
	if (output > max_out)
	{
		output = max_out;
	}
	else if (output < min_out)
	{
		output = min_out;
	}
	return output;
}
