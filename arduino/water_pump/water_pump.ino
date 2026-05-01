#include <Arduino.h>
#define DO_PIN 9
#define DO_PIN_2 11

unsigned long pump1OffAt = 0;
unsigned long pump2OffAt = 0;
const unsigned long PULSE_MS = 1000; // each trigger keeps pump ON for 4s

void triggerPump(uint8_t pin, unsigned long &offAt)
{
	digitalWrite(pin, HIGH);
	offAt = millis() + PULSE_MS;
}

void setup()
{
	pinMode(DO_PIN, OUTPUT);
	pinMode(DO_PIN_2, OUTPUT);
	Serial.begin(9600);
	digitalWrite(DO_PIN, LOW);
	digitalWrite(DO_PIN_2, LOW);
}

void loop()
{
	// Command protocol from Python:
	// "1\n" => pulse DO_PIN for PULSE_MS
	// "2\n" => pulse DO_PIN_2 for PULSE_MS
	if (Serial.available() > 0)
	{
		char cmd = Serial.read();
		if (cmd == '1')
		{
			triggerPump(DO_PIN, pump1OffAt);
		}
		else if (cmd == '2')
		{
			triggerPump(DO_PIN_2, pump2OffAt);
		}
	}

	unsigned long now = millis();
	if (pump1OffAt != 0 && (long)(now - pump1OffAt) >= 0)
	{
		digitalWrite(DO_PIN, LOW);
		pump1OffAt = 0;
	}
	if (pump2OffAt != 0 && (long)(now - pump2OffAt) >= 0)
	{
		digitalWrite(DO_PIN_2, LOW);
		pump2OffAt = 0;
	}
}