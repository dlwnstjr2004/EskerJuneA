#include <Arduino.h>
#include <SPI.h>
#include <ssd1351.h>

#define NEUTRAL 307
#define REVERSE 205
#define FORWARD 312 //409 is mad

const int SteeringPin = 4;
const int ForwardPin = 3;


int tmp_angle = 90;
int angle = 0;
int newAngle = 0;
const int MaxChars = 4;
char strValue[MaxChars+1];
int idx = 0;
int value = 0;

//====imsi
int idx_check = 0;

unsigned int SteeringPWM;
unsigned int ForwardPWM;

int map(int x, int in_min, int in_max, int out_min, int out_max) {
  int toReturn = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min ;
  return toReturn;
}

void setup()
{
  Serial.begin(115200);
  analogWriteFrequency(SteeringPin, 50);
  analogWriteFrequency(ForwardPin, 50);
  analogWriteResolution(12);
  pinMode(SteeringPin, OUTPUT);
  pinMode(ForwardPin, OUTPUT);

  SteeringPWM = NEUTRAL;
  ForwardPWM = NEUTRAL;
  
  analogWrite(SteeringPin, SteeringPWM);
  analogWrite(ForwardPin, ForwardPWM);
  delay(2500);

  angle = 90;
}

void serialEvent()
{
  while(Serial.available())
  {
    char ch = Serial.read();
    Serial.write(ch); // one letter at once,
    if (ch == '0'){ motor_stop(); }
    else if (ch == '1') { motor_level1(); }
    else if (ch == '2') { motor_level2(); }
    else if (ch == '3') { motor_level3(); }
    else if (ch == '4') { motor_level4(); }
    else if (ch == '5') { motor_level5(); }
    else if (ch == '6') { motor_level6(); }
    else Serial.write("error");
  }
}

/*
void motor_normal()
{
  analogWrite(ForwardPin, 311);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motor_slow()
{
  analogWrite(ForwardPin, 307);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
*/

void motor_stop()
{
  analogWrite(ForwardPin, 307);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motor_level1()
{
  analogWrite(ForwardPin, 308);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motor_level2()
{
  analogWrite(ForwardPin, 309);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motor_level3()
{
  analogWrite(ForwardPin, 310);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motor_level4()
{
  analogWrite(ForwardPin, 311);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motor_level5()
{
  analogWrite(ForwardPin, 312);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motor_level6()
{
  analogWrite(ForwardPin, 313);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}



void loop()
{
  serialEvent();
}
