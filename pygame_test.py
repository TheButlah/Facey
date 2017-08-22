import pygame
import pygame.camera
import time
import numpy as np
import matplotlib.pyplot as plt

class Buffer:

    def __init__(self, intial_array):
        self.oldest_index = 0
        self.size = intial_array.shape[0]
        self.array = intial_array

    def add_image(self, image):
        self.array[self.oldest_index] = image
        self.oldest_index = (self.oldest_index + 1) % self.size

def main():

    buffer = Buffer(np.zeros((10,480,640,3)))
    size = (640,480)
    display = pygame.display.set_mode(size,0)
    pygame.camera.init()
    cam = pygame.camera.Camera(pygame.camera.list_cameras()[0], size)
    cam.start()
    snapshot = pygame.surface.Surface(size, 0, display)

    try:
        t0 = time.time()
        while True:
            if cam.query_image():
                snapshot = cam.get_image(snapshot)
                display.blit(snapshot, (0, 0))
                pygame.display.flip()
                if time.time() - t0 > 0.1:
                    img = cam.get_image()
                    raw = img.get_buffer().raw
                    arr = np.flip(np.frombuffer(raw, dtype=np.ubyte).reshape(480, 640, 3), 2)
                    buffer.add_image(arr)

    except KeyboardInterrupt:
        pass

    cam.stop()

    plt.imshow(buffer.array[-1])
    plt.show()

if __name__ == '__main__':
    main()

