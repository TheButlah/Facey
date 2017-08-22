import pygame
import pygame.camera
import numpy as np
import matplotlib.pyplot as plt


def main():
    pygame.camera.init()
    cam = pygame.camera.Camera("/dev/video0", (640, 480))
    cam.start()
    img = cam.get_image()
    cam.stop()
    raw = img.get_buffer().raw
    arr = np.flip(np.frombuffer(raw, dtype=np.ubyte).reshape(480, 640, 3), axis=2)
    plt.imshow(arr)
    plt.draw()

if __name__ == '__main__':
    main()
