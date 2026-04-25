import picar_4wd as fc
import sys
import tty
import termios
import asyncio
import time
import select

power_val = 60
key = 'status'
print("If you want to quit.Please press q")
def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ready =select.select([sys.stdin],[],[],0.1)[0]
        if ready:
            ch = sys.stdin.read(1)
        else:
            ch='\x00'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def readkey(getchar_fn=None):
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)

def Keyborad_control():
    global power_val
    moving=True
    time.sleep(1)
    
    while True:
        key=readkey()
        if key=='6':
            if power_val <=90:
                power_val += 10
                print("power_val:",power_val)
        elif key=='4':
            if power_val >=10:
                power_val -= 10
                print("power_val:",power_val)
        if key=='w':
            fc.forward(power_val)
            moving=True
        elif key=='a':
            fc.turn_left(power_val)
            moving=True
        elif key=='s':
            fc.backward(power_val)
            moving=True
        elif key=='d':
            fc.turn_right(power_val)
            moving=True
        elif key =='\x00':
            if moving:
                fc.stop()
                moving = False
        else:
            fc.stop()
            moving =False
        if key=='q':
            print("quit") 
            fc.stop() 
            break  
if __name__ == '__main__':
    Keyborad_control()