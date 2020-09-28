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
int newSpeed = 0;
const int MaxChars = 2;
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
    String inString = Serial.readStringUntil('e');
    //Serial.print(inString);
    int index1 = inString.indexOf(','); 
    int index2 = inString.length();
    String SpeedValue = inString.substring(0, index1);
    String AngleValue = inString.substring(index1+1,index2);
    Serial.print("Speed:");
    Serial.println(SpeedValue);
    Serial.print("Angle:");
    Serial.println(AngleValue);

    //motor_level(SpeedValue.toInt());
    steering(AngleValue.toInt());
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

void motor_stop()
{
  analogWrite(ForwardPin, 307);
}
void motor_level1()
{
  analogWrite(ForwardPin, 308);
}
void motor_level2()
{
  analogWrite(ForwardPin, 309);
}
void motor_level3()
{
  analogWrite(ForwardPin, 310);
}
void motor_level4()
{
  analogWrite(ForwardPin, 311);
}
void motor_level5()
{
  analogWrite(ForwardPin, 312);
}
void motor_level6()
{
  analogWrite(ForwardPin, 313);
}
*/

void motor_level(int value)
{
  //307 is stop
  int power = 304+value;
  analogWrite(ForwardPin, power);
}

void steering(int value)
{
  //from 45 ~ 135 degree
  value = value + 90;
  analogWrite(SteeringPin, map(value, 0, 180, 205, 409));
}


void loop()
{
  serialEvent();
  analogWrite(ForwardPin, 307);
}
