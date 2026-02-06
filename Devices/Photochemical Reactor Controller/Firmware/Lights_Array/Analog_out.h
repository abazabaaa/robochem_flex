#ifndef ANALOG_OUT_H
#define ANALOG_OUT_H

/**
 * Definitions for controlling a 0-10v analog output.
 *
 * Author: Simone Pilon <s.pilon at uva.nl> - Noël Research Group - 2024
 */

#include "PID.h"
#include <Arduino.h>

#define ANALOGOUT_ERROR_OK 0		// No error

#define DEFAULT_CALIBRATION 2.10


/**
 * Interface for AnalogOut analog voltage signal generation.
 * Each interface outputs an analog value to a pwm pin, then reads the actual value from an
 * analog pin.
 */
class AnalogOut
{
	public:

		/**
		 * Constructor.
		 */
		AnalogOut(uint8_t pwm_out_pin, uint8_t analog_in_pin);

		/**
		 * Intialize pin modes and values.
		 */
		void init();

		/**
		 * Current signal setpoint.
		 *
		 * @return The signal setpoint.
		 */
		uint8_t get_setpoint();

		/**
		 * Read current signal.
		 *
		 * @return The signal measured value.
		 */
		uint8_t get_signal();

		/**
		 * Change signal setpoint.
		 *
		 * @parameter value The desired signal value.
		 */
		void set_setpoint(uint8_t value);

		/**
		 * Read input value and adjust output to match setpoint. 
		 */
		void update();

		/**
		 * Set PID Proportional constant.
		 *
		 * @param p PID proportional constant.
		 */
		void set_P(float p);

		/**
		 * Get PID Proportional constant.
		 *
		 * @return PID proportional constant.
		 */
		float get_P();

		/**
		 * Set PID integral constant.
		 *
		 * @param i PID integral constant.
		 */
		void set_I(float i);

		/**
		 * Get PID integral constant.
		 *
		 * @return PID integral constant.
		 */
		float get_I();

		/**
		 * Set PID derivative constant.
		 *
		 * @param d PID derivative constant.
		 */
		void set_D(float d);

		/**
		 * Get PID derivative constant.
		 *
		 * @return PID derivative constant.
		 */
		float get_D();


		float calibration;	/**< Calibration value to convert % to raw value. */

		static uint8_t error;	/**< Error code of last operation. */

	protected:

		bool active;			/**< True if the output is active (non zero). */
		uint8_t output_pin;	/**< PWM output pin. */
		uint8_t input_pin;	/**< Analog input pin. */
		PID pid_control;		/**< PID controller for value. */
};

#endif //  ANALOG_OUT_H
