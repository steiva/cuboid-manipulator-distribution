from screeninfo import get_monitors
import cv2
from threading import Thread
import sys  
sys.path.insert(0, '../cuboid-manipulator/')
from cuboid_manipulator import core, utils
import dearpygui.dearpygui as dpg
import numpy as np
from pynput import keyboard
import time
from paths import PROJECT_ROOT
import logging
import string

CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.json'
MASK_PATH = PROJECT_ROOT / 'config' / 'mask.npy'

logging.basicConfig(filename=f'{PROJECT_ROOT}/logs/temp_log.log', filemode='w',
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%I:%M:%S %p', level = logging.DEBUG)

def get_prim_monitor_wh() -> tuple:
    for m in get_monitors():
        if m.is_primary:
            return m.width, m.height
        
def frame_to_gpu_format(frame):
    data = cv2.resize(frame, (1200, 900))
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = data.ravel()
    data = np.asarray(data, dtype='f')
    texture_data = np.true_divide(data, 255.0)
    return texture_data
        
class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self):
        self.stream = None
        self.available_sources = []
        self.width = 2592
        self.height = 1944
        self.find_video_sources()
        self.cam_mtx = core.read_config(['camera_params', 'camera_2', '2592 x 1944', 'cam_mtx'])
        self.dist = core.read_config(['camera_params', 'camera_2', '2592 x 1944', 'dist'])
        self.new_cam = core.read_config(['camera_params', 'camera_2', '2592 x 1944', 'newcam_mtx'])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def test_video_device(self, src):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            print('Warning: unable to open video source: ', src)
            return None
        else:
            return cap
        
    def find_video_sources(self):
        for src in range(-1, 10):
            stream = self.test_video_device(src)
            if stream is not None:
                stream.set(cv2.CAP_PROP_FRAME_WIDTH,self.width)
                stream.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height)
                stream.set(cv2.CAP_PROP_FPS, 60) 
                stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G')) 
                self.available_sources.append(stream)
        if self.available_sources:
            self.stream = self.available_sources[0]
        else:
            print('Warning: Video source not found, setting to empty stream ...')
            self.stream = cv2.VideoCapture()

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                ret, frame = self.stream.read()
                if ret:
                    frame = cv2.undistort(frame, self.cam_mtx, self.dist, None, self.new_cam)
                else:
                    continue
                self.grabbed, self.frame = ret, frame

    def stop(self):
        self.stopped = True

class DragToSelect():

    def __init__(self, well_plate) -> None:
        self.well_plate = well_plate
        self.init_pos = None
        self.final_pos = None
        self.stack = []

    def draw_box(self):
        if dpg.is_item_focused('well_plate_selector'):
            x,y = dpg.get_drawing_mouse_pos()
            if not self.stack:
                self.init_pos = [x,y]
            else:
                prev_rect = self.stack.pop()
                dpg.delete_item(prev_rect)
            rect = dpg.draw_rectangle(self.init_pos, [x,y], parent='well_plate_selector', color=(255,255,255,100))
            self.stack.append(rect)

    def clear_all(self):
        self.final_pos = dpg.get_drawing_mouse_pos()
        if self.stack:
            rect = self.stack.pop()
            dpg.delete_item(rect)
            self.well_plate.well_plate_region_select(self.init_pos, self.final_pos)

class PickingProcedure():

    def __init__(self, logger) -> None:
        self.running = False
        self.paused = False
        self.logger = logger
        self.dash = None
        self.move = None
        self.well_indexes = None
        self.min_dist_thresh = None
        self.failure_thresh = None
        self.z_offset = None
        self.z_deposit = None
        self.min_size = None
        self.max_size = None
        self.rotation_zero = None
        self.rotation_max = None
        self.rotation_speed = None
        self.rotation_accel = None
        self.fail_wait_time = None
        self.dest_wait_time = None
        self.grid = None
        self.cuboid_choice = None
        self.draw_previous = False
        self.draw_choice = False
        self.well_plate_type = None
        self.index = 0
        self.anchors = None
        self.calibration_z = None
        self.tf_mtx = None
        self.proc = None

    def init_robot_pos(self) -> None:
        self.move.JointMovJ(-30,0,0,self.rotation_zero)
        self.move.Sync()
        self.dash.SpeedJ(100)
        self.dash.AccJ(100)

    def re_init_config(self) -> None:
        self.anchors = core.read_config(['configs', 'anchors'])
        self.calibration_z = np.mean(self.anchors[:,2])
        self.tf_mtx = core.read_config(['configs', 'tf_mtx'])

    def on_press(self, key, abortKey='esc', pauseKey = 'p'):
        try:
            k = key.char  # single-char keys
        except:
            k = key.name  # other keys

        if k == pauseKey:
            self.paused = not(self.paused)
            if self.paused:
                self.logger.log_info('Pressed p, pausing experiment ...')
            else:
                self.logger.log_info('Pressed p, unpausing experiment ...')

        # print('pressed %s' % (k))
        if k == abortKey:
            self.logger.log_info('Pressed esc, ending loop ...')
            self.running = False
            self.paused = False
            return False  # stop listener
        
    def loop(self):
        self.running = True
        #-----------------Robot initialization------------------
        self.dash.ResetRobot()
        self.dash.ClearError()
        self.dash.EnableRobot()

        self.dash.SpeedJ(self.rotation_speed)
        self.dash.AccJ(self.rotation_accel)
        self.dash.SpeedL(100)
        self.dash.AccL(100)

        logging.info('Starting automatic picking procedure ...')

        self.move.JointMovJ(-30,0,0,120)
        self.move.Sync()
        #-------------------------------------------------------
        if self.well_plate_type == '96':
            width = 8
            height = 12
        elif self.well_plate_type == '384':
            width = 16
            height = 24
        elif self.well_plate_type == '24':
            width = 4
            height = 6

        if self.well_indexes:
            flat_indexes = [int(well[0] + well[1]*width) for well in self.well_indexes]
        else:
            flat_indexes = range(width*height)
            self.well_indexes = []
            for row in range(height):
                row_indices = []
                for col in range(width):
                    index = (row+1, col+1)
                    row_indices.append(index)
                self.well_indexes += row_indices

        flat_indexes = sorted(flat_indexes)

        #-----------------Main loop-----------------------------
        while self.index < len(flat_indexes):
            if self.running == False:
                break
            if self.paused == True:
                while True:
                    time.sleep(0.1)
                    if self.paused == False:
                        break

            correct_size_cuboids = self.proc.correct_size_cuboids
            pickable_cuboids = self.proc.pickable_cuboids

            if correct_size_cuboids is not None:
                if len(correct_size_cuboids) == 0:
                    self.logger.log_warning(f'Not enough cuboids of necessary size, breaking loop at index {self.index} ...')
                    logging.warning(f'Not enough cuboids of necessary size, breaking loop at index {self.index} ...')
                    break

            if pickable_cuboids is not None:
                if len(pickable_cuboids) == 0:
                    self.logger.log_warning('No more pickable cuboids, shake dish. Pausing program ...')
                    logging.warning('No more pickable cuboids, shake dish. Pausing program ...')
                    self.paused = True
                    continue
                else:
                    x_message,y_message = self.well_indexes[self.index]
                    alphabet_list = list(string.ascii_uppercase)[:width]
                    alphabet_list.reverse()
                    x_message = alphabet_list[int(x_message)]
                    y_message = int(y_message + 1)
                    message = f'IDX = {self.index}| Filling {x_message}{y_message} ({flat_indexes[self.index]}). Pickable cuboids: {len(pickable_cuboids)}/{len(correct_size_cuboids)}'
                    self.logger.log_info(message)
                    logging.info(message)
                    self.cuboid_choice = pickable_cuboids.iloc[np.random.randint(len(pickable_cuboids))]
                    self.draw_previous = False
                    self.draw_choice = True

            
            X, Y, _ = self.tf_mtx @ (self.cuboid_choice.cX, self.cuboid_choice.cY, 1)
            #----------------Starting move sequence------------------
            # 1. Cuboid pickup:
            self.move.MovL(X, Y, 0, self.rotation_zero)
            self.move.Sync()
            utils.correct_J4_angle(self.rotation_zero, self.dash, self.move)
            self.move.RelMovL(0,0, self.calibration_z + self.z_offset)
            self.move.Sync()
            utils.correct_J4_angle(self.rotation_max, self.dash, self.move)
            self.move.RelMovL(0,0, -self.calibration_z - self.z_offset)
            self.move.Sync()

            # 2. Go to position above well plate but don't deposit yet:
            grid_x, grid_y = self.grid[flat_indexes[self.index]] # for well plate grid
            # grid_x, grid_y, self.deposit_z = [300,-125, -36,  0][:3]
            self.move.MovL(grid_x, grid_y, 0, self.rotation_max)
            self.move.Sync()
            utils.correct_J4_angle(self.rotation_max, self.dash, self.move)

            self.draw_choice = False
            self.draw_previous = True
            cuboid_df = self.proc.correct_size_cuboids

            # Check if robot missed target:
            prev_x, prev_y = self.cuboid_choice.cX, self.cuboid_choice.cY
            distances = cuboid_df.apply(lambda row: np.sqrt((row['cX'] - prev_x)**2 + (row['cY'] - prev_y)**2), axis = 1).to_numpy()

            if any(distances < self.failure_thresh):
                missed_indexes = np.where(distances < self.failure_thresh)[0]
                missed_cuboids = cuboid_df.iloc[missed_indexes]
                if any(missed_cuboids.pickable):
                    self.logger.log_warning('Miss detected, retrying ...')
                    logging.warning('Miss detected, retrying ...')

                    self.move.MovL(X, Y, 0, self.rotation_max)
                    self.move.Sync()
                    utils.correct_J4_angle(self.rotation_max, self.dash, self.move)
                    # self.move.RelMovL(0,0, self.calibration_z + self.z_offset)
                    self.move.RelMovL(0,0, self.calibration_z + 1)
                    self.move.Sync()
                    utils.correct_J4_angle(self.rotation_zero, self.dash, self.move)
                    time.sleep(self.fail_wait_time)
                    self.move.RelMovL(0,0, -self.calibration_z - self.z_offset)
                    self.move.Sync()
                    self.move.RelMovJ(-30,0,0,0)
                    self.move.Sync()
                    time.sleep(0.5)
                    continue

            self.move.RelMovL(0,0, self.z_deposit)
            self.move.Sync()

            interm_angle = self.rotation_zero + int((self.rotation_max - self.rotation_zero)/2)
            # utils.correct_J4_angle(interm_angle, self.dash, self.move)
            utils.correct_J4_angle(self.rotation_zero, self.dash, self.move)
            time.sleep(self.dest_wait_time)
            self.move.RelMovL(0,0, -self.z_deposit)
            self.move.Sync()
                
            self.index += 1
            self.draw_previous = False

            if self.running == False:
                break

        self.logger.log('Picking finished! Press any key to exit ...')
        logging.info('Picking finished! Press any key to exit ...')

    def create_listener(self):
        return keyboard.Listener(on_press=self.on_press)

    def execute(self):
        self.listener = self.create_listener()
        self.listener.start()  # start to listen on a separate thread
        Thread(target=self.loop, args=(), daemon=False).start()
        self.listener.join() # wait for abortKey

class Processing:

    def __init__(self, callbacks, cr):
        self.cr = cr
        self.pp = None
        self.callbacks = callbacks
        # self.dash = callbacks.dash
        # self.move = callbacks.move
        self.frame = None
        self.result = None
        self.busy = False
        #-------Histogram----------
        self.max_hist_value = 0
        self.counter = 0
        self.avg_hist = None
        #--------------------------
        self.pickup_range = [250, 450]
        self.laser_coor = None
        self.anchor_mask = None
        self.avg_triang_area = 0
        self.total_triang_area = 0
        self.triang_counter = 0

        self.correct_size_cuboids = None
        self.pickable_cuboids = None

    def start(self, func):
        self.thread = Thread(target=func, args=())
        self.thread.start()
        return self
    
    def video_stream_test(self):
        self.busy = True
        self.result = self.frame
        self.busy = False

    def get_triang(self):
        self.busy = True

        frame = cv2.bilateralFilter(self.frame, 5, 175, 175)
        plot_img = frame.copy()
        triangle = core.get_triangle(frame)
        cv2.drawContours(plot_img, triangle, -1,(0,255,0),2)
        if triangle:
            triangle_area = cv2.contourArea(triangle[0])
            self.total_triang_area += triangle_area
            self.triang_counter += 1
            self.avg_triang_area = self.total_triang_area/self.triang_counter
            cv2.putText(plot_img, f"Visual reference area: {triangle_area}", (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.putText(plot_img, f"Visual reference mean area: {self.avg_triang_area}", (25,70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        self.result = plot_img
        self.busy = False

    def get_anchor_regions(self):
        self.busy = True
        plot_img = self.frame.copy()
        self.anchor_mask, anchor_regions = core.find_anchor_regions(self.frame)
        detected = cv2.drawContours(plot_img, anchor_regions, -1, color=(0, 255, 0), thickness=2)
        self.result = detected
        self.busy = False

    def cuboid_recognition(self):
        self.busy = True

        frame = self.frame
        self.cr.preprocess_frame(frame)
        self.cr.locked = self.callbacks.detection_region_locked
        
        plot_img = frame.copy()
        
        if not self.cr.locked:
            self.cr.get_circles(frame)

        if self.cr.best_circ is None:
            print('no circle found')
            return
        a,b,r = self.cr.best_circ

        cv2.circle(plot_img, (a, b), r, (0, 0, 255), 2)
        cv2.circle(plot_img, (a, b), r-self.cr.initial_offset, (0, 255, 255), 2)
        cv2.circle(plot_img, (a, b), r-self.cr.initial_offset-self.cr.pickup_offset, (0, 255, 0), 2)
        self.cr.find_contours(frame, offset=self.cr.initial_offset)

        self.cr.cuboid_dataframe(self.cr.cuboids, 25)

        pickable = self.cr.cuboid_df.loc[self.cr.cuboid_df.pickable == True]
        not_pickable = self.cr.cuboid_df.loc[self.cr.cuboid_df.pickable == False]

        if pickable is not None and self.callbacks.analysis_window_ready:
            # print(dpg.get_item_info('hist_plot'))
            # print(list(cr.cuboid_df.diam_microns.values))
            hist_data = list(pickable.diam_microns.values)
            hist_values = np.histogram(hist_data, bins = 120, range = (0,600))[0]
            max_hist_value = np.max(hist_values)
            if max_hist_value > self.max_hist_value:
                self.max_hist_value = max_hist_value
            if self.avg_hist is None:
                self.avg_hist = hist_values

            curr_avg_hist = (self.avg_hist*self.counter + hist_values)/(self.counter + 1)
            self.counter += 1
            self.avg_hist = curr_avg_hist

            dpg.set_axis_limits('y_axis', 0, self.max_hist_value)
            dpg.set_value('hist_plot', [hist_data])
            dpg.set_value('avg_hist_plot', [list(np.arange(2.5,605-2.5,5)),list(self.avg_hist)])
            # dpg.set_value('avg_hist_plot', [[100,200],[300,400]])

        # clusters = cr.cuboid_df.loc[cr.cuboid_df.poly_approx > 14]
        cv2.drawContours(plot_img,pickable.contour.to_numpy(), -1,(0,255,0),1)
        cv2.drawContours(plot_img,not_pickable.contour.to_numpy(), -1,(0,255,255),1)
        # cv2.drawContours(plot_img,clusters.contour.to_numpy(), -1,(0,0,255),2)
        cv2.putText(plot_img, f"Found: {len(self.cr.cuboid_df)}", (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) 

        if self.cr.selected:
            cv2.drawContours(plot_img, self.cr.selected, -1,(255,0,0),2)
        self.result = plot_img 
        self.busy = False

    def find_and_draw_laser(self):
        self.busy = True
        frame = self.frame
        plot_img = frame.copy()
        mask = np.load(MASK_PATH)
        cX, cY = core.laser_finder(frame, mask)
        self.laser_coor = (cX, cY)
        plot_img = np.bitwise_or(plot_img, cv2.dilate(cv2.Canny(mask,100,200), np.ones((3,3),np.uint8))[:,:,np.newaxis])
        cv2.circle(plot_img, (cX, cY), 5, (255, 0, 0), 2)
        cv2.circle(plot_img, (cX, cY), 25, (0, 0, 255), 2)
        cv2.circle(plot_img, (cX, cY), 50, (0, 0, 255), 2)
        self.result = plot_img
        self.busy = False

    def process_frame(self):
            self.cr.preprocess_frame(self.frame)
            self.cr.find_contours(self.frame, offset=self.cr.initial_offset)
            self.cr.cuboid_dataframe(self.cr.cuboids, 25)
            cb_df = self.cr.cuboid_df
            self.correct_size_cuboids = cb_df.loc[(cb_df.diam_microns > self.pickup_range[0]) &
                                            (cb_df.diam_microns < self.pickup_range[1]) &
                                            (cb_df.pickable == True) &
                                            (cb_df.circularity < 0.95)] #Temp fix for improper bubble recognition.
            self.pickable_cuboids = self.correct_size_cuboids.loc[(self.correct_size_cuboids.min_dist > self.pp.min_dist_thresh)]

    def draw_picking_region(self, frame):
        a,b,r = self.cr.best_circ
        cv2.circle(frame, (a, b), r, (0, 0, 255), 2)
        cv2.circle(frame, (a, b), r - self.cr.initial_offset, (0, 255, 255), 2)
        cv2.circle(frame, (a, b), r - self.cr.initial_offset - self.cr.pickup_offset, (0, 255, 0), 2)

    def draw_cuboid_contours(self, frame):
        pickable = self.cr.cuboid_df.loc[self.cr.cuboid_df.pickable == True]
        not_pickable = self.cr.cuboid_df.loc[self.cr.cuboid_df.pickable == False]
        bubble = self.cr.cuboid_df.loc[self.cr.cuboid_df.bubble == False]
        cv2.drawContours(frame, pickable.contour.to_numpy(), -1,(0,255,255),2)
        cv2.drawContours(frame, not_pickable.contour.to_numpy(), -1,(0,0,255),2)
        cv2.drawContours(frame, bubble.contour.to_numpy(), -1,(0,0,255),2)
        cv2.drawContours(frame, self.pickable_cuboids.contour.to_numpy(), -1,(0,255,0),2)
        if self.pp.cuboid_choice is not None and self.pp.draw_choice:
            cv2.circle(frame, (self.pp.cuboid_choice.cX, self.pp.cuboid_choice.cY), 15, (255, 0, 0), 2)
            cv2.circle(frame, (self.pp.cuboid_choice.cX, self.pp.cuboid_choice.cY), int(self.pp.min_dist_thresh), (100, 100, 0), 2)
        if self.pp.cuboid_choice is not None and self.pp.draw_previous:
            cv2.circle(frame, (self.pp.cuboid_choice.cX, self.pp.cuboid_choice.cY), self.pp.failure_thresh, (0, 0, 0), 2)
            # cv2.rectangle(frame, (self.pp.cuboid_choice.cX - self.pp.failure_thresh, self.pp.cuboid_choice.cY - self.pp.failure_thresh),
            # (self.pp.cuboid_choice.cX + self.pp.failure_thresh, self.pp.cuboid_choice.cY + self.pp.failure_thresh), (0, 0, 0), 2)

    def picking_procedure_processing(self):

        self.busy = True
        # start_time = time.time()
        frame = self.frame
        plot_img = frame.copy()
        self.process_frame()
        # end_time = time.time()
        # print(f'Processing time: {end_time - start_time}')
        self.draw_picking_region(plot_img)
        self.draw_cuboid_contours(plot_img)
        self.result = plot_img
        self.busy = False
