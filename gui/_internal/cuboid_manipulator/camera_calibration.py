import numpy as np
import cv2 as cv
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
# Input the chessboard parameters,
chessboardSize = (8, 6)
#frameSize = (1600, 1200)
frameSize = (2592,1944)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0],
                       0:chessboardSize[1]].T.reshape(-1, 2)


# Input chess square size.
size_of_chessboard_squares_mm = 8
objp = objp * size_of_chessboard_squares_mm

objpoints = []
imgpoints = []

images = sorted(glob.glob('camera_data/raw/*.png'))
#print(len(images))
for idx, image in enumerate(images):

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('image', img)
    # Find the chess board corners
    # If the findChessboardCorners() doesn't work well for you, try the
    # findChessboardCornersSB() alternative. Sometimes it proves to be more
    # robust.
    ret, corners = cv.findChessboardCornersSB(gray, chessboardSize, None)
    #print(corners)
    # Check if the algorithm detected any chessboard in the image.
    # If either one of the images gives false, the pair will not be considered
    # for recognition of the corners.
    print(f'i = {idx}, res = {ret}')

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(cornersL)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, cornersL, ret)
        cv.namedWindow('Calibration',  cv.WINDOW_NORMAL) # creating a GUI window cv2 is a module called open cv which has all the methods related to computer vision
        cv.resizeWindow('Calibration', 1348, 1011)
        cv.imshow('Calibration', img)
        cv.waitKey(2000)
    idx += 1
cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None)
height, width, channelsL = img.shape
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
    cameraMatrix, dist, (width, height), 1, (width, height))


print("Camera Matrix")
print(cameraMatrix)
np.save('cam_matrices/cam_mtx_1944.npy', cameraMatrix)

print("Distortion Coeff")
print(dist)
np.save('cam_matrices/dist_1944.npy', dist)

print("Region of Interest")
print(roi)
np.save('cam_matrices/roi_1944.npy', roi)

print("New Camera Matrix")
print(newCameraMatrix)
np.save('cam_matrices/newcam_mtx_1944.npy', newCameraMatrix)

# img = cv.imread('data/image0.png')
# dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
#cv.imwrite('calibresult.png', dst)