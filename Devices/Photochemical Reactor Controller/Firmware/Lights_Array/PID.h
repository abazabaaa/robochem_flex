#ifndef PID_H
#define PID_H

/**
 * Simple PID controller implementation.
 *
 * Author: Simone Pilon <s.pilon at uva.nl> - Noël Research Group - 2024
 */

#include <Arduino.h>

/**
 * PID controller.
 */
class PID
{
	public:

		/**
		 * Constructor.
		 */
		PID();

		/**
		 * Change setpoint.
		 * Resets the controller.
		 *
		 * @parameter setpoint The desired input value.
		 */
		void set_setpoint(float setpoint);

		/**
		 * Get setpoint.
		 *
		 * @return The setpoint.
		 */
		float get_setpoint();

		/**
		 * Compute new output value based on input.
		 * Should be called at regular intervals.
		 *
		 * @parameter input The current input value.
		 * @return the calculated output value.
		 */
		float output(float input);

		float kp;	/**< PID Proportional coefficient. */
		float ki;	/**< PID integral coefficient. */
		float kd;	/**< PID derivative coefficient. */

		float max_out;	/**< Maximum value for output. */
		float min_out;	/**< Minimum value for output. */
	protected:

		float setpoint;	/**< Desired signal intensity. */
		float last_input;	/**< Last recorded input value. */
		float sum_out;	/**< Sum of PID integral component. */
};

#endif //  PID_H
