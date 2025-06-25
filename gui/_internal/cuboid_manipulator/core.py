import cv2
import numpy as np
import pandas as pd
import logging
from scipy.spatial import KDTree
import threading
from cuboid_manipulator.dobot_api import DobotApiDashboard, DobotApiMove
import time 
from cuboid_manipulator.utils import assign_corners
import json
from paths import PROJECT_ROOT
import os

CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.json'
MASK_PATH = PROJECT_ROOT / 'config' / 'mask.npy'

camera_res_dict = {
            '240':(320,240),
            '480':(640,480),
            '600':(800,600),
            '768':(1024, 768),
            '960':(1280, 960),
            '1200':(1600, 1200),
            '1536':(2048, 1536),
            '1944':(2592, 1944),
            '2448':(3264, 2448),
            '2160': (3840, 2160)}

# class ShowFrame(threading.Thread):
#     def __init__(self, frame=None, window_name = None, name='show-frame-thread'):
#         self.window_name = window_name
#         self.frame = frame
#         self.stopped = False
#         super(ShowFrame, self).__init__(name=name)
#         self.start()

#     def start(self):
#         threading.Thread(target=self.show, args=()).start()
#         return self
    
#     def show(self):
#         while not self.stopped:
#             cv2.imshow(self.window_name, self.frame)
#             if cv2.waitKey(1) == ord("q"):
#                 self.stopped = True

#     def stop(self):
#         self.stopped = True

class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if ret:
                self.last_frame = frame

    def read(self):
        if self.last_frame is not None:
            return True, self.last_frame
        else: 
            return False, None
        
    def release(self):
        self.camera.release()


class Camera():
    def __init__(self, index = 0, no_buffer = True):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,2592)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1944) 
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        self.no_buffer = no_buffer
        if self.no_buffer:
            logging.info('Using camera without buffer ...')
            self.cap = CameraBufferCleanerThread(self.cap)
        logging.info('Camera initialized ...')

        # self.cam_mtx = np.load('../config/cam_mtx_1944.npy')
        # self.dist = np.load('../config/dist_1944.npy')
        # self.new_cam = np.load('../config/newcam_mtx_1944.npy')

        # Res : 2592 x 1944 on the new camera.
        # self.cam_mtx = np.load('../cuboid-manipulator/cam_matrices/cam_mtx_1994.npy')
        # self.dist = np.load('../cuboid-manipulator/cam_matrices/dist_1994.npy')
        # self.new_cam = np.load('../cuboid-manipulator/cam_matrices/newcam_mtx_1994.npy')
        self.cam_mtx = read_config(['camera_params', 'camera_2', '2592 x 1944', 'cam_mtx'])
        self.dist = read_config(['camera_params', 'camera_2', '2592 x 1944', 'dist'])
        self.new_cam = read_config(['camera_params', 'camera_2', '2592 x 1944', 'newcam_mtx'])

        # Res : 3264 x 2448
        # self.cam_mtx = np.load('../cuboid-manipulator/cam_matrices/cam_mtx_2448.npy')
        # self.dist = np.load('../cuboid-manipulator/cam_matrices/dist_2448.npy')
        # self.new_cam = np.load('../cuboid-manipulator/cam_matrices/newcam_mtx_2448.npy')

        # Nikon D5600
        
        # self.cam_mtx = np.load('../cuboid-manipulator/cam_matrices/cam2_mtx_1944.npy')
        # self.dist = np.load('../cuboid-manipulator/cam_matrices/dist2_1944.npy')
        # self.new_cam = np.load('../cuboid-manipulator/cam_matrices/newcam2_mtx_1944.npy')

        self.window_name = 'frame'

    def get_frame(self, undist: bool = True, gray: bool = False) -> np.ndarray:
        """
        Function for reading frames from capture with direct undistortion
        and settings for getting grayscale images directly.

        Args:
            undist (bool, optional): Boolean wether to undistort the image or not.
            Defaults to True.
            gray (bool, optional): Boolean wether to give grayscale images directly.
            Defaults to True.

        Returns:
            np.ndarray: A captured frame represented by a numpy array.
        """
        if self.cap.read() is not None:         
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_frame()
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if undist:
                frame = cv2.undistort(frame, self.cam_mtx, self.dist, None, self.new_cam)
            return frame
        else:
            self.get_frame()

    def get_window(self) -> None:
        """
        Function for opening a window that fits the screen properly
        for 1920x1080 resolution monitor.
        """        
        cv2.namedWindow(self.window_name,  cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1348, 1011)

    def release_camera(self) -> None:
        """
        Release camera capture.
        """        
        logging.info('Releasing capture ...')
        self.cap.release()

class Core():
    def __init__(self) -> None:
        self.cuboids = None
        self.cuboid_df = None
        self.selected = []
        self.best_circ = None
        self.pickup_offset = 30
        self.initial_offset = 40
        self.locked = False
        self.size_conversion_ratio = read_config(['configs','size_conversion_ratio'])

    def reload_inits(self) -> None:
        self.size_conversion_ratio = read_config(['configs','size_conversion_ratio'])

    def preprocess_frame(self, frame: np.ndarray) -> None:
        self.gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bil_fr = cv2.bilateralFilter(self.gray_fr, 5, 175, 175)
        self.thresh_fr = self.threshold_frame(self.gray_fr)
        self.bubble_fr = self.bubble_frame(self.gray_fr)

    def threshold_frame(self, frame: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(frame,(11,11),0) # added blur
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,5)
        return thresh 
    
    def bubble_frame(self, frame: np.ndarray) -> np.ndarray:
        thresh = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,29,5) # For cuboids

        # blur = cv2.GaussianBlur(frame,(11,11),0) # added blur
        # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,5) # For discoids
        return thresh

    def find_contours(self, frame: np.ndarray, offset: int = 0) -> None:
        """
        General contour finding pipeline of objects inside a circular contour. In our case
        we are looking for objects in a Petri dish.

        Args:
            frame (np.ndarray): frame taken from camera cap, or just an image.
            offset (int, optional): offset for radius of the circle where cuboids
            are detected. Offset is counted inwards. Defaults to 0.
        """
        self.preprocess_frame(frame)
        if len(frame.shape) == 3: #ensure frame is grayscale by looking at frame shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # frame = cv2.bilateralFilter(frame, 5, 175, 175)
        thresh = self.threshold_frame(frame)
        kernel = np.ones((3,3),np.uint8)
        res = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        result = mask_frame(res, self.best_circ, offset)
        # Find all the contours in the resulting image.
        contours, hierarchy = cv2.findContours(
            result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # convex_contours = []
        # for contour in contours:
        #     if cv2.isContourConvex(contour):
        #         convex_contours.append(contour)
        
        self.cuboids = contours

    def find_contours_v2(self, frame: np.ndarray, offset: int = 0) -> None:
        if len(frame.shape) == 3: #ensure frame is grayscale by looking at frame shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        masked_gray = mask_frame(frame, self.best_circ, offset)
        new_image = cv2.convertScaleAbs(masked_gray, alpha=1.5, beta=0.5)

        gamma = 2
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(new_image, lookUpTable)

        arr = res.ravel()
        # Assuming you have an array named 'arr'
        filtered_arr = arr[arr != 0]

        low_bound = np.where(np.histogram(filtered_arr, bins=256, range=(0, 256))[0]>10000)[0][0]
        _, thresholded_img = cv2.threshold(res, low_bound-5, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresholded_img, 0, 255)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        self.cuboids = contours



    def get_circles(self, frame: np.ndarray) -> None:
        """
        Function looks for a petri dish in the frame, and assigns the smallest one
        to a class variable for storage. 

        Args:
            frame (np.ndarray): frame in which to detect the petri dish.
        """
        if len(frame.shape) == 3: #ensure frame is grayscale by looking at frame shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        blur = cv2.GaussianBlur(frame,(3,3),0)
        # if not self.inv:
        #     ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # else:
        ret, thresh = cv2.threshold(blur,125,255,cv2.THRESH_BINARY_INV)

        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(thresh,kernel,iterations = 3)

        blur2 = cv2.blur(dilation, (7, 7))
        detected_circles = cv2.HoughCircles(image=blur2,
                                            method=cv2.HOUGH_GRADIENT,
                                            dp=1.2,
                                            minDist=500,
                                            param1=100,
                                            param2=50,
                                            minRadius=700,
                                            maxRadius=900
                                            )

        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))

            pt = detected_circles[0][0]
            a, b, r = pt
            
            if self.best_circ is None or r < self.best_circ[2]:
                self.best_circ = pt

            best_center = np.array(self.best_circ[:2])
            curr_center = np.array([a, b])
            if np.sqrt(np.sum((best_center - curr_center)**2)) > 30:
                self.best_circ = pt

    def locate_dish(self, frame: np.ndarray) -> None:

        if len(frame.shape) == 3: #ensure frame is grayscale by looking at frame shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       
        new_image = cv2.convertScaleAbs(frame, alpha=1.5, beta=0.5)

        gamma = 2
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(new_image, lookUpTable)

        thresh = cv2.adaptiveThreshold(res,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,29,5)
        im_floodfill = thresh.copy()
        h, w = thresh.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (thresh.shape[0]//2,thresh.shape[1]//2), 255) 
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = thresh | im_floodfill_inv
        blob = cv2.bitwise_not(im_out)
        dilated = cv2.morphologyEx(blob, cv2.MORPH_DILATE, np.ones((9,9),np.uint8), iterations=3) 

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours is not None:
            if len(contours) > 1:
                print("Warning: More than one contour detected")

            M = cv2.moments(contours[0])
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            centroid = (centroid_x, centroid_y)
            x, y, w, h = cv2.boundingRect(contours[0])
            r = int(np.mean([w, h]) / 2)

            if (self.best_circ is None or r < self.best_circ[2]) and r < frame.shape[0]//2:
                self.best_circ = (centroid_x, centroid_y, r)

            # if self.best_circ is not None:
            #     best_center = np.array(self.best_circ[:2])
            #     curr_center = np.array([centroid_x, centroid_y])
            #     if np.sqrt(np.sum((best_center - curr_center)**2)) > 50:
            #         self.best_circ = (centroid_x, centroid_y, r)

    def size_conversion(self, cuboid_size_px: float) -> float:
        """Size conversion from pixels to microns and then to 
        cuboid diameter. In this calculation, cuboids are assumed to 
        look circular. If we regard the cuboids as squares (top down view),
        just a square root of cuboid_size_micron2 should be sufficient.

        Args:
            cuboid_size_px (float): cuboid size in pixels as seen by the camera.

        Returns:
            float: cuboid diameter in microns.
        """    
        cuboid_size_micron2 = cuboid_size_px * self.size_conversion_ratio * 1000000
        cuboid_diameter = 2 * np.sqrt(cuboid_size_micron2 / np.pi)
        return cuboid_diameter

    def cuboid_dataframe(self, contours: list, filter_thresh: int = None) -> None:
        """
        Function creates dataframe with all necessary information about the cuboids:
        The area of the individual cuboids, the coordinates of their center, distance
        to closest neighbor, and a boolean status if it is pickable or not, based on 
        wether it is located in a pickable region.

        Args:
            contours (list): a list of detected contours.
            filter_thresh (int): filter out the contours based on size. For example,
            if filter_thresh = 10, all contours that are smaller than 10 are filtered out.
        """        
        df_columns = ['contour', 'area', 'cX', 'cY', 'min_dist', 'pickable', 'diam_microns', 'circularity', 'bubble']
        if not contours:
            self.cuboid_df = pd.DataFrame(columns=df_columns)
            return
        cuboid_df = pd.DataFrame({'contour':contours})
        cuboid_df['area'] = cuboid_df.apply(lambda row : cv2.contourArea(row.iloc[0]), axis=1)
        if filter_thresh:
            cuboid_df = cuboid_df.loc[cuboid_df.area > filter_thresh]
            if len(cuboid_df) == 0:
                self.cuboid_df = pd.DataFrame(columns=df_columns)
                return
        centers = cuboid_df.apply(lambda row : self.contour_centers([row.iloc[0]])[0], axis=1)
        cuboid_df[['cX','cY']] = pd.DataFrame(centers.tolist(),index=cuboid_df.index)

        cuboid_df.dropna(inplace=True)

        T = KDTree(cuboid_df[['cX', 'cY']].to_numpy())
        cuboid_df['min_dist'] = cuboid_df.apply(lambda row: T.query((row.iloc[2], row.iloc[3]), k = 2)[0][-1], axis = 1)

        # cuboid_df['poly_approx'] = cuboid_df.apply(lambda row: len(cv2.approxPolyDP(row[0], 0.01*cv2.arcLength(row[0], True), True)), axis = 1)
        def calculate_circularity(contour):
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter == 0:
                return 0
            return (4 * np.pi * area) / (perimeter ** 2)

        cuboid_df['circularity'] = cuboid_df.apply(lambda row: calculate_circularity(row.iloc[0]), axis=1)

        def determine_pickable(row):
            cub = np.array((row.iloc[2], row.iloc[3]))
            contour = row.iloc[0]
            center = np.array(self.best_circ[:2])

            not_too_far = np.linalg.norm(cub-center) < self.best_circ[-1]-self.initial_offset-self.pickup_offset
            bubble = bool(self.bubble_fr[int(cub[1]), int(cub[0])])
            return bubble and not_too_far
        
        def bubble(row):
            cub = np.array((row.iloc[2], row.iloc[3]))
            bubble = bool(self.bubble_fr[int(cub[1]), int(cub[0])])
            return bubble
        
        def not_too_far(row):
            cub = np.array((row.iloc[2], row.iloc[3]))
            center = np.array(self.best_circ[:2])
            not_too_far = np.linalg.norm(cub-center) < self.best_circ[-1]-self.initial_offset-self.pickup_offset
            return not_too_far

        cuboid_df['pickable'] = cuboid_df.apply(determine_pickable, axis = 1)
        cuboid_df['bubble'] = cuboid_df.apply(bubble, axis = 1)
        # cuboid_df['not_too_far'] = cuboid_df.apply(not_too_far, axis = 1)

        cuboid_df['diam_microns'] = cuboid_df.apply(lambda row: self.size_conversion(row.iloc[1]), axis = 1)
        self.cuboid_df = cuboid_df

    def mousecallback(self,event,x,y,flags,param):
        """
        OpenCV mouse callback function for registering double clicks.
        In our case used to select individual cuboids and pick them insted of
        picking everything automatically. If there is another double click then
        deselect the cuboid.

        Args:
            event (int?): OpenCV event code. Usually an int?
            x (int?): x position of the click in the image.
            y (int?): y position of the click in the image.
            flags (_type_): No idea.
            param (_type_): Optional parameter the can be returned?
        """        
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.selected:
                for contour in self.selected:
                    r=cv2.pointPolygonTest(contour, (x,y), False)
                    if r > 0:
                        self.selected.remove(contour)
                        return

            for contour in self.cuboids:
                r=cv2.pointPolygonTest(contour, (x,y), False)
                if r>0:
                    self.selected.append(contour)

    def raw_mousecallback(self,x,y):
        if self.selected:
            for contour in self.selected:
                r=cv2.pointPolygonTest(contour, (x,y), False)
                if r > 0:
                    self.selected.remove(contour)
                    return
        for contour in self.cuboids:
            r=cv2.pointPolygonTest(contour, (x,y), False)
            if r>0:
                self.selected.append(contour)

    def contour_centers(self, contours: tuple) -> list:
        """
        Function calculates the centers of the inputed contours.

        Args:
            contours (tuple): A tuple of contours to be filtered, normally outputed 
            by cv2.findContours() function.

        Returns:
            list: outputs list of coordinates of the contour centers.
        """
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
            else:
                centers.append((None,None))
        return centers
    
def get_triangle(frame: np.ndarray, area_thresh: int = 10000) -> list:
    """Function looks for a triangle in a frame. The trianlge is an
    easy target to find and is used for size conversion from pixels to mm.

    Args:
        frame (np.ndarray): frame captured by the camera
        area_thresh (int, optional): Minimum area for the triangle. Defaults to 10000.

    Returns:
        list: returns list of contours (should be just one though).
    """
    if len(frame.shape) == 3: #check if frame is already grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(frame,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(thresh, kernel)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    triangle = []
    for cont in contours:
        epsilon = 0.01*cv2.arcLength(cont,True)
        approx = cv2.approxPolyDP(cont,epsilon,True)
        if len(approx) == 3 and cv2.contourArea(cont) > area_thresh:
            triangle.append(cont) 

    return triangle

def get_triangle_area(cap: Camera) -> float:
    """Function observes a triangle in a video feed,
    getting its mean area for size conversion. 

    Args:
        cap (Camera): current opened camera class.

    Returns:
        float: mean area of the triangle.
    """    
    cap.get_window()
    all_areas = []
    while True:
        frame = cap.get_frame()
        frame = cv2.bilateralFilter(frame, 5, 175, 175)
        plot_img = frame.copy()
        triangle = get_triangle(frame)
        cv2.drawContours(plot_img, triangle, -1,(0,255,0),2)
        if triangle:
            triangle_area = cv2.contourArea(triangle[0])
            all_areas.append(triangle_area)
            cv2.putText(plot_img, f"Visual reference area: {triangle_area}", (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.putText(plot_img, f"Mean visual reference area: {np.mean(all_areas)}", (25,90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        cv2.imshow(cap.window_name, plot_img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            print('Image saved!')
            cv2.imwrite('../outputs/paper figures/size_conversion.jpg', plot_img)

    cv2.destroyAllWindows()
    mean_area = np.mean(all_areas)
    return mean_area
    
def find_anchor_regions(frame: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Find black regions on the robot platform. The regions are ordinary pieces 
    of black tape that the laser uses for anchors. This function is needed to
    create a mask for those regions of black tape so that the camera <-> robot
    calibration is more precise.

    Args:
        frame (np.ndarray): single frame from camera to find anchor regions on.

    Returns:
        tuple[np.ndarray, list]: returns the mask as well as the contours for plotting.
    """    
    if len(frame.shape) == 3: #check if frame is already grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(frame,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    kernel = np.ones((7,7),np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    img_erosion = cv2.erode(morph, kernel, iterations=10)
    contours, _ = cv2.findContours(img_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cont in contours:
        epsilon = 0.1*cv2.arcLength(cont,True)
        approx = cv2.approxPolyDP(cont,epsilon,True)
        if len(approx) == 4:
            squares.append(cont)
    mask = np.zeros(frame.shape[0:2], dtype='uint8')
    mask = cv2.drawContours(mask, squares, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    return (mask, squares)

def laser_finder(frame: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """
    This function is used to find a bright spot of the laser on the image.
    It will work if a correct mask is supplied (produced by finding anchor regions).
    TODO: incorporate old method in case mask is not supplied. The current thresholding is static,
    things change under different lighting and this method is not robust.

    Args:
        frame (np.ndarray): image provided by the camera.
        mask (np.ndarray): a mask that filters out anything but the black anchor regions.

    Returns:
        tuple[int, int]: coordinates of the laser center.
    """    

    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.bitwise_and(frame,frame,mask = mask)
    ret, thresh = cv2.threshold(frame,230,255,cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        return (None, None)

    return (cX, cY)

def robot_2_camera_calibration(dash: DobotApiDashboard, move: DobotApiMove, cap: Camera) -> list:
    """Pipeline for robot <=> camera calibration.

    Args:
        dash (DobotApiDashboard): MG400 robot API for info.
        move (DobotApiMove): MG400 robot API for movement.
        cap (core.Camera): current capture class.

    Returns:
        list: list of recorded pixel coords.
        These coords are needed to produce a transformation matrix.
    """    
    # anchors = np.load('../config/anchors.npy')
    anchors = read_config(['configs','anchors'])
    # mask = np.load('../config/mask.npy')
    mask = np.load(MASK_PATH)
    recorded = []

    dash.ClearError()
    dash.EnableRobot()

    logging.info('='*10 + ' Robot <=> Camera calibration initiated ' + '='*10)
    cap.get_window()
    for idx, anchor in enumerate(anchors):
        x,y,z,r = anchor
        move.MovL(x,y,z,r)
        move.Sync()

        time.sleep(0.3)
        frame = cap.get_frame()
        plot_img = frame.copy()

        cX, cY = laser_finder(frame, mask)
        plot_img = np.bitwise_or(plot_img, cv2.dilate(cv2.Canny(mask,100,200), np.ones((3,3),np.uint8))[:,:,np.newaxis])
        cv2.circle(plot_img, (cX, cY), 5, (255, 0, 0), 2)
        cv2.circle(plot_img, (cX, cY), 25, (0, 0, 255), 2)
        recorded.append((cX, cY))
            
        cv2.imshow(cap.window_name, plot_img) 
        cv2.waitKey(0)
        cv2.imwrite(f'../outputs/paper figures/calibration_{idx}.jpg', plot_img)

    cv2.destroyAllWindows()
    return recorded

def write_tf_mtx(recorded_coords: list) -> None:
    """Function writes the transformation matrix for the robot <=> camera calibration.

    Args:
        recorded_coords (list): pixel coordinates of the laser point in 4 anchor regions.
    """    
    # anchors = np.load('../config/anchors.npy')
    anchors = read_config(['configs','anchors'])
    for coord in recorded_coords:
        if None in coord:
            logging.warning('Found NoneType in recorded coordinates. Redo.')
            print('Found NoneType in recorded coordinates. Redo.')
            return

    xys = [(arr[0], arr[1]) for arr in anchors]
    robot_coor = assign_corners(xys, reverse=True) # assign corners to the robot coordinates at the 4 corner positions 
    pix_coor = assign_corners(recorded_coords)

    features_mm_to_pixels_dict = {} # setting up an empty dictionary to store the mapping of the corners from coordinate to pixel
    for key, value in robot_coor.items():
        features_mm_to_pixels_dict[value] = pix_coor[key]

    tf_mtx = compute_tf_mtx(features_mm_to_pixels_dict) # method of cv_core module that calculates transformation matrix
    logging.info('Writing transformation matrix to disk ...')
    write_config(['configs','tf_mtx'], tf_mtx)
    # np.save('../config/tfm_mtx.npy', tf_mtx) 

def pipette_positioning(cap: Camera, cr: Core, filter_thresh: int = None) -> None:
    """
    Function launches window where contours can be selected manually.
    This function would usually be used for initial pipette positioning.

    Args:
        cap (Camera): current opened camera class.
        cr (Core): Core class.
        filter_thresh (int): threshold argument passed to cuboid_dataframe function (see docstring).
    """    
    cap.get_window()
    def on_change(val): pass
    cv2.createTrackbar('Manual Lock', cap.window_name, int(cr.locked), 1, on_change)
    cv2.setMouseCallback(cap.window_name, cr.mousecallback)
    while(True):
        frame = cap.get_frame()

        cr.preprocess_frame(frame)
        
        plot_img = frame.copy()

        cr.locked = bool(cv2.getTrackbarPos('Manual Lock', cap.window_name))
        
        if not cr.locked:
            cr.get_circles(frame)
            # cr.locate_dish(frame)

        if cr.best_circ is None:
            print('no circle found')
            continue
        
        a,b,r = cr.best_circ

        cv2.circle(plot_img, (a, b), r, (0, 0, 255), 2)
        cv2.circle(plot_img, (a, b), r-cr.initial_offset, (0, 255, 255), 2)
        cv2.circle(plot_img, (a, b), r-cr.initial_offset-cr.pickup_offset, (0, 255, 0), 2)
        cr.find_contours(frame, offset=cr.initial_offset)

        if filter_thresh:
            cr.cuboid_dataframe(cr.cuboids, filter_thresh)
        else:
            cr.cuboid_dataframe(cr.cuboids)

        pickable = cr.cuboid_df.loc[cr.cuboid_df.pickable == True]
        not_pickable = cr.cuboid_df.loc[cr.cuboid_df.pickable == False]
        correct_size_cuboids = cr.cuboid_df.loc[(cr.cuboid_df.diam_microns > 350) &
                                          (cr.cuboid_df.diam_microns < 450) &
                                          (cr.cuboid_df.pickable == True)]
        good_boids = correct_size_cuboids.loc[(correct_size_cuboids.min_dist > 50)]
        # clusters = cr.cuboid_df.loc[cr.cuboid_df.poly_approx > 14]
        cv2.drawContours(plot_img,not_pickable.contour.to_numpy(), -1,(0,0,255),2)
        cv2.drawContours(plot_img,pickable.contour.to_numpy(), -1,(0,255,255),2)
        cv2.drawContours(plot_img,good_boids.contour.to_numpy(), -1,(0,255,0),2)
        # cv2.drawContours(plot_img,clusters.contour.to_numpy(), -1,(0,0,255),2)
        cv2.putText(plot_img, f"Found: {len(cr.cuboid_df)}", (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) 

        if cr.selected:
            cv2.drawContours(plot_img, cr.selected, -1,(255,0,0),2) 

        cv2.imshow(cap.window_name, plot_img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('../outputs/paper figures/recognized_cuboids.png', plot_img)
            print("Image saved!")

    cv2.destroyAllWindows()

def write_anchor_regions(cap: Camera, cr: Core) -> None:
    """
    Function detects anchor regions in video stream and saves anchor mask to disk.
    Doesn't need to be used each run, but rather only after physical changes in the setup.

    Args:
        cap (Camera): current opened camera class.
        cr (Core): Core class.
    """    
    cap.get_window()
    while(True):
        frame = cap.get_frame()
        plot_img = frame.copy()
        mask, anchor_regions = find_anchor_regions(frame)
        detected = cv2.drawContours(plot_img, anchor_regions, -1, color=(0, 255, 0), thickness=2)

        cv2.imshow(cap.window_name, detected)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            print('Image saved!')
            cv2.imwrite('../outputs/paper figures/anchor_regions.jpg', detected)

    cv2.destroyAllWindows()
    # np.save('../config/mask.npy', mask)
    np.save(MASK_PATH, mask)

def mask_frame(frame: np.ndarray, pt: tuple, offset: int) -> np.ndarray:
    """Function creates a circular mask and applies it to an image. In our case this is
    used to select the area in the petri dish only and find contours there.

    Args:
        frame (np.ndarray): frame that needs to be masked.
        pt (tuple): circle parameters, center coordinates a,b and radius r.
        offset (int): an offset for mask application. Useful if circle is too large.

    Returns:
        np.ndarray: returns a masked image.
    """
    a, b, r = pt
    # Create mask to isolate the information in the petri dish.
    mask = np.zeros_like(frame)
    mask = cv2.circle(mask, (a, b), r-offset, (255, 255, 255), -1)
    # Apply the mask to the image.
    result = cv2.bitwise_and(frame.astype('uint8'), mask.astype('uint8'))
    return result

def compute_tf_mtx(mm2pix_dict: dict) -> np.ndarray:
    """Function computes the transformation matrix between real-world
    coordinates and pixel coordinates in an image.

    Args:
        mm2pix_dict (dict): Dictionary mapping real-world coordinates
        to pixel coordinates. Example for four points:
        {(382.76, -113.37): (499, 412),
        (225.27, 94.68): (240, 103),
        (386.5, 91.55): (492, 98),
        (221.25, -110.62): (248, 419)}

    Returns:
        np.ndarray: array that represents the transformation matrix.
    """
    A = np.zeros((2 * len(mm2pix_dict), 6), dtype=float)
    b = np.zeros((2 * len(mm2pix_dict), 1), dtype=float)
    index = 0
    for XY, xy in mm2pix_dict.items():
        X = XY[0]
        Y = XY[1]
        x = xy[0]
        y = xy[1]
        A[2 * index, 0] = x
        A[2 * index, 1] = y
        A[2 * index, 2] = 1
        A[2 * index + 1, 3] = x
        A[2 * index + 1, 4] = y
        A[2 * index + 1, 5] = 1
        b[2 * index, 0] = X
        b[2 * index + 1, 0] = Y
        index += 1
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    tf_mtx = np.zeros((3, 3))
    tf_mtx[0, :] = np.squeeze(x[:3])
    tf_mtx[1, :] = np.squeeze(x[3:])
    tf_mtx[-1, -1] = 1
    return tf_mtx

def write_config(property_path: list, value) -> None:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    if type(value) is np.ndarray:
        value = value.tolist()

    query = config
    for p in property_path[:-1]:
        query = query[p]

    query[property_path[-1]] = value

    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

def read_config(property_path: list):
    with open(CONFIG_PATH, 'r') as f:

        config = json.load(f)

    query = config
    for p in property_path:
        if p not in query:
            return None
        query = query[p]

    if type(query) is list:
        query = np.array(query)

    return query

def check_config_exists(property_path: list) -> bool:
    query = read_config(property_path)
    if query is None:
        return False
    elif type(query) is np.ndarray:
            if query.size == 0:
                return False
            else:
                return True
    else:
         return True
    
def check_well_plate_calibration_exists(well_plate_type: str) -> bool:
    return os.path.isfile(f'{PROJECT_ROOT}/config/well_plate_{well_plate_type}.npy')

# def write_config(property_name: str, value) -> None:
#     with open('../config/config.json', 'r') as f:
#         config = json.load(f)

#     if type(value) is np.ndarray:
#         value = value.tolist()

#     config['configs'][property_name] = value
    
#     with open('../config/config.json', 'w') as f:
#         json.dump(config, f, indent=4)

# def read_config(property_name: str):
#     with open('../config/config.json', 'r') as f:
#         config = json.load(f)

#     value = config['configs'][property_name]

#     if type(value) is list:
#         value = np.array(value)

#     return value