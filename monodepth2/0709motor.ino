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
      strValue[idx++] = ch; //마지막을 의미하는 문자 s나 a가 들어있을 것.
      idx_check = idx;
      Serial.print("strValue Test: ");
      Serial.print(strValue[0]);
      Serial.print(strValue[1]);
      Serial.print(strValue[2]);
      Serial.print(strValue[3]);
      Serial.println(strValue[4]);

      Serial.print("newAngle Test:");
      
      if(strValue[idx_check-1]=="a"){// accelerate이라면
        newSpeed = strValue[0]-48
        if ((newSpeed >= 0)&&(newSpeed <= 6)){ motor_level(newSpeed); }
        else Serial.print("ErrorCode:0002 speed variant is strange!");
      }
      else if(strValue[idx_check-1]=="s"){ //steering이라면
        idx_check--;
        if(idx_check == 4){
          newAngle = (strValue[0]-48)*pow(10,(idx_check-1)) + (strValue[1]-48)*pow(10,(idx_check-2)) + (strValue[3]-48)*pow(10,(idx_check-3)) + (strValue[4]-48)*pow(10,(idx_check-4));
        }
        else if(idx_check == 3){
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

        newAngle = newAngle/5+26; // 5pixel당 1degree 단순 선형적으로 계산, 320일 때 90degree : no steering
        analogWrite(SteeringPin, map(newAngle, 0, 180, 205, 409));
        Serial.println(map(newAngle, 0, 180, 205, 409));
      }
      else{ Serial.print("ErrorCode:0001 no flags"); }
      idx_check = 0;
      idx = 0;
    }
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
  int power = 307+value;
  analogWrite(ForwardPin, power);
}


void loop()
{
  serialEvent();
}
