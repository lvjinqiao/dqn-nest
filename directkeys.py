import ctypes
import time
import  pyautogui
SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
I = 0x17
J = 0x24
K = 0x25
U=0x16


PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def go_left():
    PressKey(A)
    time.sleep(0.2)
    ReleaseKey(A)

def go_right():
    PressKey(D)
    time.sleep(0.2)
    ReleaseKey(D)

def go_up():
    PressKey(W)
    time.sleep(0.2)
    ReleaseKey(W)

def go_down():
    PressKey(S)
    time.sleep(0.2)
    ReleaseKey(S)

def attack_one():
    PressKey(U)
    time.sleep(0.2)
    ReleaseKey(U)


def attack_two():
    PressKey(J)
    time.sleep(0.2)
    ReleaseKey(J)

def attack_three():
    PressKey(I)
    time.sleep(0.2)
    ReleaseKey(I)

def attack_four():
    PressKey(K)
    time.sleep(0.2)
    ReleaseKey(K)

def pres_skll_one():
    PressKey(D)
    time.sleep(0.05)
    ReleaseKey(D)
    PressKey(S)
    time.sleep(0.05)
    ReleaseKey(S)
    PressKey(A)
    time.sleep(0.05)
    ReleaseKey(A)
    PressKey(I)
    time.sleep(0.05)
    ReleaseKey(I)



def pres_skll_two():
    PressKey(D)
    time.sleep(0.05)
    ReleaseKey(D)


    PressKey(S)
    time.sleep(0.05)
    ReleaseKey(S)

    PressKey(D)
    time.sleep(0.05)
    ReleaseKey(D)

    # PressKey(S)
    # time.sleep(0.05)
    # ReleaseKey(S)

    PressKey(J)
    time.sleep(0.05)
    ReleaseKey(J)


def pres_skll_three():
    PressKey(S)
    time.sleep(0.05)



    PressKey(D)
    time.sleep(0.05)
    ReleaseKey(S)

    # PressKey(S)
    # time.sleep(0.05)
    # ReleaseKey(S)
    #
    # PressKey(D)
    # time.sleep(0.05)
    # ReleaseKey(D)

    PressKey(U)
    time.sleep(0.05)

    ReleaseKey(U)
    ReleaseKey(D)


def pres_skll_four():
    PressKey(S)
    time.sleep(0.05)
    ReleaseKey(S)

    PressKey(A)
    time.sleep(0.05)
    ReleaseKey(A)

    PressKey(S)
    time.sleep(0.05)
    ReleaseKey(S)

    PressKey(D)
    time.sleep(0.05)
    ReleaseKey(D)

    PressKey(U)
    time.sleep(0.05)
    ReleaseKey(U)

def pres_skll_five():
    PressKey(A)
    time.sleep(0.05)
    ReleaseKey(A)
    PressKey(S)
    time.sleep(0.05)
    ReleaseKey(S)
    PressKey(D)
    time.sleep(0.05)
    ReleaseKey(D)
    PressKey(I)
    time.sleep(0.05)
    ReleaseKey(I)



def pres_skll_six():
    PressKey(A)
    time.sleep(0.05)
    ReleaseKey(A)


    PressKey(S)
    time.sleep(0.05)
    ReleaseKey(S)

    PressKey(A)
    time.sleep(0.05)
    ReleaseKey(A)

    # PressKey(S)
    # time.sleep(0.05)
    # ReleaseKey(S)

    PressKey(J)
    time.sleep(0.05)
    ReleaseKey(J)


def pres_skll_seven():
    PressKey(S)
    time.sleep(0.05)



    PressKey(A)
    time.sleep(0.05)
    ReleaseKey(S)

    # PressKey(S)
    # time.sleep(0.05)
    # ReleaseKey(S)
    #
    # PressKey(D)
    # time.sleep(0.05)
    # ReleaseKey(D)

    PressKey(U)
    time.sleep(0.05)

    ReleaseKey(U)
    ReleaseKey(A)


def pres_skll_eight():
    PressKey(S)
    time.sleep(0.05)
    ReleaseKey(S)

    PressKey(D)
    time.sleep(0.05)
    ReleaseKey(D)

    PressKey(S)
    time.sleep(0.05)
    ReleaseKey(S)

    PressKey(A)
    time.sleep(0.05)
    ReleaseKey(A)

    PressKey(U)
    time.sleep(0.05)
    ReleaseKey(U)




def pres_skll_nine():
    PressKey(S)
    time.sleep(0.08)
    ReleaseKey(S)

    time.sleep(0.02396)

    PressKey(D)
    time.sleep(0.03)
    ReleaseKey(D)
    time.sleep(0.02396)

    PressKey(S)
    time.sleep(0.08)
    ReleaseKey(S)
    time.sleep(0.02396)

    PressKey(D)
    time.sleep(0.03)
    ReleaseKey(D)

    time.sleep(0.02396)

    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)


if __name__ == '__main__':

    PressKey(W)
    ReleaseKey(W)
