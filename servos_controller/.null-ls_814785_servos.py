from pyfirmata import Arduino, util
import time

board = Arduino('/dev/tty.usbmodem13301')

servo_x_pin = board.get_pin('d:9:s')
servo_y_pin = board.get_pin('d:8:s')

def move_servo(angle, servo_pin):
    if 0 <= angle <= 180:
        servo_pin.write(angle)
        print(f"Servo moved to {angle} degrees")
    else:
        print("Angle should be between 0 and 180 degrees.")

try:
    while True:
        move_servo(0, servo_x_pin)
        move_servo(0, servo_y_pin)
        time.sleep(2) 
        move_servo(90, servo_x_pin)
        move_servo(90, servo_y_pin)
        time.sleep(2)
        move_servo(180, servo_x_pin)
        move_servo(180, servo_y_pin)
        time.sleep(2)
except KeyboardInterrupt:
    print("Arrêt manuel du programme")
finally:
    board.exit()
