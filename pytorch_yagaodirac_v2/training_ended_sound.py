import winsound
import time
import math

__all__ = ['play_noise']

def play_noise():
    print("playing noise~~~~~~~~~")
    freq_base = 220
    small_two = math.pow(2.,1./12)
    while True:
        for i in range(25):
            freq = freq_base*math.pow(small_two,i)
            #print(freq)
            winsound.Beep(int(freq), 100)
        time.sleep(1) # Wait for 1 second
