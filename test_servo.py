import maestro
import time
servo = maestro.Controller()
servo.setTarget(0,6000)  #set servo to move to center position
servo.setTarget(1,6000)
print("done")
servo.close
