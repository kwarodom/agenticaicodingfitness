from djitellopy import Tello
import time

t = Tello()
t.connect()
battery = t.get_battery()
print(f'Battery: {battery}%')

if battery < 30:
    print('Battery too low for tricks! Charge the drone first.')
    exit()

try:
    t.takeoff()
    time.sleep(2)

    # Rise up for visibility
    print('>> Rising to performance altitude')
    t.move_up(80)
    time.sleep(1)

    # Forward-back dash
    print('>> Speed dash forward & back')
    t.move_forward(100)
    time.sleep(0.5)
    t.move_back(100)
    time.sleep(1)

    # Flip combo: forward then back
    print('>> Front flip!')
    t.flip_forward()
    time.sleep(1.5)

    print('>> Back flip!')
    t.flip_back()
    time.sleep(1.5)

    # Side-to-side strafing with flips
    print('>> Strafe left + left flip!')
    t.move_left(60)
    t.flip_left()
    time.sleep(1.5)

    print('>> Strafe right + right flip!')
    t.move_right(120)
    t.flip_right()
    time.sleep(1.5)

    # Return to center
    t.move_left(60)
    time.sleep(1)

    # 360 spin
    print('>> 360 spin!')
    t.rotate_clockwise(360)
    time.sleep(1.5)

    # Square patrol pattern
    print('>> Square patrol')
    for _ in range(4):
        t.move_forward(60)
        t.rotate_clockwise(90)
        time.sleep(0.5)

    # Grand finale: rise + flip + spin + descend
    print('>> Grand finale!')
    t.move_up(50)
    t.flip_forward()
    time.sleep(1)
    t.rotate_counter_clockwise(360)
    time.sleep(1)

    # Bow (gentle dip)
    print('>> Take a bow!')
    t.move_down(30)
    time.sleep(0.5)
    t.move_up(30)
    time.sleep(1)

    print('>> Landing!')
    t.land()

except Exception as e:
    print(f'Error during flight: {e}')
    t.land()
finally:
    t.end()