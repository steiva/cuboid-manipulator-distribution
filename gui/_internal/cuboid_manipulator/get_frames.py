import cv2
from cuboid_manipulator import core

# Create videoCapture objects for both video streams.
cap = cv2.VideoCapture(0)
#cap = cv_core.set_res(cap, (2592, 1944))
cap = core.set_res(cap, core.camera_res_dict['1944'])
num = 0

# Set infinite loop to capture images from video.
while True:
    # We use the .grab() method to reduce the lag between the two videos.
    ret, frame = cap.read()
    cv2.namedWindow('Image',  cv2.WINDOW_NORMAL) # creating a GUI window cv2 is a module called open cv which has all the methods related to computer vision
    cv2.resizeWindow('Image', 1348, 1011)
    cv2.imshow('Image', frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        # Put whatever directory is convenient for you.
        cv2.imwrite('camera_data/raw/image' +
                    str(num) + '.png', frame)
        # cv2.imwrite('image'+
        #     str(num) + '.png', frame)
        print("images saved!")
        num += 1

cap.release()
cv2.destroyAllWindows()