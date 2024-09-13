import numpy as np
import time
import pyautogui
import sounddevice as sd
import soundfile as sf
import cv2


class BoundingBoxWidget(object):
    def __init__(self):
        self.original_image = cv2.imread('screen.jpg')
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordinates on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print('top left: {}, bottom right: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            print('x,y,w,h : ({}, {}, {}, {})'.format(self.image_coordinates[0][0], self.image_coordinates[0][1],
                                                      self.image_coordinates[1][0] - self.image_coordinates[0][0],
                                                      self.image_coordinates[1][1] - self.image_coordinates[0][1]))

            # Draw rectangle
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    def get_coords(self):
        return self.image_coordinates


# compute MSE between two images
def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mean_se = err/(float(h*w))
   return mean_se, diff


def warn():
    filename = "error.wav"  # Replace with the path to your own sound file
    data, _ = sf.read(filename, dtype='float32')
    sd.play(data, samplerate=44100)
    sd.wait()


def main():
    time.sleep(6)

    image = pyautogui.screenshot()
    ss = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cv2.imwrite('screen.jpg', ss)

    boundingbox_widget = BoundingBoxWidget()
    while True:
        cv2.imshow('image', boundingbox_widget.show_image())
        key = cv2.waitKey(1)

        if key == ord('x'):
            cv2.destroyAllWindows()
            break 

    coords = boundingbox_widget.get_coords()
    print(coords)
    user_coords = [coords[0][0], coords[0][1]]
    nxt_x = coords[1][0] - coords[0][0]
    nxt_y = coords[1][1] - coords[0][1]
    user_coords.append(nxt_x)
    user_coords.append(nxt_y)
    win_region = (user_coords[0], user_coords[1], user_coords[2], user_coords[3])

    # win_region = (228, 869, 107, 670)  # <-- this is my local bounding box

    # initial screen
    prev_screen = pyautogui.screenshot(region=win_region)

    while True:
        # take ss
        curr_screen = pyautogui.screenshot(region=win_region)

        prev_screen.save('previous.jpg')
        curr_screen.save('current.jpg')

        im1 = cv2.imread('previous.jpg')
        im2 = cv2.imread('current.jpg')

        # convert the images to grayscale
        img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        error, diff = mse(img1, img2)
        print("matching err between the two images:", error)

        threshold_err = 0.5

        if error >= threshold_err:
            warn()

        # update prev
        prev_screen = curr_screen

        time.sleep(1)


if __name__ == "__main__":
    main()
