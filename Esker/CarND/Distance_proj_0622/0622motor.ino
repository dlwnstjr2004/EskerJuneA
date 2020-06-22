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
    if (ch=='1'){ motor_normal(); }
    else if (ch == '0') { motor_slow(); }
    else Serial.write("error");
  }
}


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

void loop()
{
  serialEvent();
}
