import sys  
sys.path.insert(0, '../config/')
# from alarm_servo import alarm_servo_list
# from alarm_controller import alarm_controller_list
import json
import numpy as np
from pynput import keyboard
import math
import collections

modes_dict = {1: 'ROBOT_MODE_INIT',
2:	'ROBOT_MODE_BRAKE_OPEN', 	
4:	'ROBOT_MODE_DISABLED',
5:	'ROBOT_MODE_ENABLE',
6:	'ROBOT_MODE_BACKDRIVE',	
7:	'ROBOT_MODE_RUNNING',
8:	'ROBOT_MODE_RECORDING',	
9:	'ROBOT_MODE_ERROR',
10: 'ROBOT_MODE_PAUSE',
11: 'ROBOT_MODE_JOG'}

def get_pose(dash, verbose = True, angle = False) -> np.ndarray:
    """Get the current arm position in format X,Y,Z,r.

    Args:
        dash (DobotApiDashboard): Dashboard class object currently 
        connected to the robot.

    Returns:
        np.ndarray: Numpy array with 4 etries: X,Y,Z,r.
    """
    if angle:
        resp = dash.GetAngle()
    else:
        resp = dash.GetPose()    
    coords = resp.split('{')[1].split('}')[0].split(',')
    coords = [float(coord) for coord in coords[:4]]
    if verbose:
        print(f'X = {coords[0]}\nY = {coords[1]}\nZ = {coords[2]}\nr = {coords[3]}')
    return np.array(coords)

class Keyboard():

    def __init__(self, dash):
        self.coords = []
        self.dash = dash
    
    def on_press(self, key):
        try:
            if key.char == 's':
                print('Position saved!')
                self.coords.append(get_pose(self.dash, verbose = False))
        except AttributeError:
            print('Special key pressed: {0}'.format(key))

    def on_release(self, key):
        if key == keyboard.Key.esc:
            return False

    def execute(self):
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()

class ManualMove():

    def __init__(self, move, dash):
        self.move = move
        self.dash = dash
        self.status = False
        self.coords = []

    def get_pose(self, dash, verbose = True) -> np.ndarray:
        """Get the current arm position in format X,Y,Z,r.

        Args:
            dash (DobotApiDashboard): Dashboard class object currently 
            connected to the robot.

        Returns:
            np.ndarray: Numpy array with 4 etries: X,Y,Z,r.
        """
        resp = dash.GetPose()    
        coords = resp.split('{')[1].split('}')[0].split(',')
        coords = [float(coord) for coord in coords[:4]]
        if verbose:
            print(f'X = {coords[0]}\nY = {coords[1]}\nZ = {coords[2]}\nr = {coords[3]}')
        return np.array(coords)
    
    def on_press(self, key):
        try:
            if key == keyboard.Key.up and not self.status:
                
                self.move.MoveJog('X-')
                self.status = True
            if key == keyboard.Key.down and not self.status:
                
                self.move.MoveJog('X+')
                self.status = True
            if key == keyboard.Key.left and not self.status:
                
                self.move.MoveJog('Y-')
                self.status = True
            if key == keyboard.Key.right and not self.status:
                
                self.move.MoveJog('Y+')
                self.status = True
            if key == keyboard.Key.page_up and not self.status:
                
                self.move.MoveJog('Z+')
                self.status = True
            if key == keyboard.Key.page_down and not self.status:

                self.move.MoveJog('Z-')
                self.status = True
            if key.char == 's':
                print('Position saved!')
                self.coords.append(self.get_pose(self.dash, False))
        except Exception as e:
            pass

    def on_release(self, key):
        self.dash.ResetRobot()
        self.status = False
        if key == keyboard.Key.esc:
            return False

    def execute(self):
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()


def assign_corners(coords, reverse = False):
    if reverse:
        offset = 1
    else:
        offset = 0

    maxx = sorted(coords, key=lambda x: x[0+offset])

    left = maxx[:2]
    right = maxx[-2:]

    l_sorty = sorted(left, key=lambda x: x[1-offset])
    r_sorty = sorted(right, key=lambda x: x[1-offset])

    ul = l_sorty[0]
    ll = l_sorty[1]

    ur = r_sorty[0]
    lr = r_sorty[1]

    corners_dict = {'ul':ul,
                    'ur':ur,
                    'lr':lr,
                    'll':ll}
    return corners_dict

def correct_J4_angle(prev_angle, dash, move):
    joint_angles = get_pose(dash, verbose=False, angle = True)
    r1, r2, r3, r4 = joint_angles
    move.JointMovJ(r1,r2,r3,prev_angle)
    move.Sync()