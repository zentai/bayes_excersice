from pynput import keyboard

def on_press(key):
    if key == keyboard.Key.shift_l:
        print("Left Shift 被按下")
    elif key == keyboard.Key.shift_r:
        print("Right Shift 被按下")

def on_release(key):
    if key == keyboard.Key.shift_l:
        print("Left Shift 释放")
    elif key == keyboard.Key.shift_r:
        print("Right Shift 释放")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

