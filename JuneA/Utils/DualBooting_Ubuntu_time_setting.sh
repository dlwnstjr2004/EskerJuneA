#!bin/bash
timedatectl set-local-rtc 1
sudo echo "UTC=no" >> /etc/default/rcS
## sudo vim /etc/default/rcS
## 추가 혹은 수정
## UTC=no
## 시각 다시맞추고 재부팅
