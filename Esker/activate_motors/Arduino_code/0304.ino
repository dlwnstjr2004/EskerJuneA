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
    if(idx < MaxChars && isDigit(ch)) {
      strValue[idx++] = ch;
      Serial.println("number detected"); // ch is number
      idx_check = idx;
    } else {
      Serial.print("strValue Test: ");
      Serial.print(strValue[0]);
      Serial.print(strValue[1]);
      Serial.print(strValue[2]);
      Serial.println(strValue[3]);

      Serial.print("newAngle Test:");
      
      if(idx_check == 3){
        newAngle = (strValue[0]-48)*pow(10,(idx_check-1)) + (strValue[1]-48)*pow(10,(idx_check-2)) + (strValue[2]-48)*pow(10,(idx_check-3));
      }
      else if(idx_check == 2){
        newAngle = (strValue[0]-48)*pow(10,(idx_check-1)) + (strValue[1]-48)*pow(10,(idx_check-2));
      }
      else if(idx_check == 1){
        newAngle = (strValue[0]-48)*pow(10,(idx_check-1));
      }
      else{
        newAngle = 0;
      }
      idx_check = 0;
      
      Serial.println(newAngle);
      
      //strValue[0] = 0;
      //strValue[1] = 0;
      //strValue[2] = 0;
      //strValue[3] = 0;
      //Serial.println("idx_check:");
      //Serial.println(idx_check);
      
      if(newAngle > 0 && newAngle < 180) {
        if(newAngle < angle){
          for(; angle > newAngle; angle--){
            tmp_angle = angle -5;
            analogWrite(SteeringPin, map(tmp_angle, 0, 180, 205, 409));
            analogWrite(ForwardPin, FORWARD);
            Serial.print("go right : ");
            Serial.println(map(tmp_angle, 0, 180, 205, 409));
          }
        }
        else{
          for(; angle < newAngle; angle++){
            tmp_angle = angle +5;
            analogWrite(SteeringPin, map(tmp_angle, 0, 180, 205, 409));
            analogWrite(ForwardPin, FORWARD);
            Serial.print("go left : ");
            Serial.println(map(tmp_angle, 0, 180, 205, 409));
          }
        }
      }
      else if(newAngle == 0){
        analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
        analogWrite(ForwardPin, NEUTRAL);
        Serial.println("Stop.");
        delay(2000);
      }
      else;
  
      idx = 0;
      angle = newAngle;
    }
  }
  //Serial.println("This serial is not available.");
}

void motortest1()
{
  analogWrite(ForwardPin, 311);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motortest2()
{
  analogWrite(ForwardPin, 310);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motortest3()
{
  analogWrite(ForwardPin, 307);
  analogWrite(SteeringPin, map(90, 0, 180, 205, 409));
}
void motortest()
{
  delay(50);
  motortest1();
  delay(50);
  motortest2();
  delay(50);
  motortest3();
  delay(50);
  motortest2();
}


void loop()
{
  serialEvent();
}
