# coding=utf8
import sys
import cv2
import numpy as np
from numpy import *
import freenect
import liblo
import random
import time
import logging
import scipy.ndimage.morphology
import argparse
import timeout_decorator

logging.basicConfig(stream = sys.stderr, level=logging.INFO)
# 1. get kinect input
# 2. bounding box calculation

parser = argparse.ArgumentParser(description='Track Motion!')
parser.add_argument('-osc', dest='osc', default=7770,help="OSC Port")
parser.set_defaults(motion=False,osc=7770)
args = parser.parse_args()

target = liblo.Address(args.osc)
def send_osc(path,*args):
    global target
    return liblo.send(target, path, *args)


screen_name = "Track Motion"

def current_time():
    return int(round(time.time() * 1000))



fullscreen = False
cv2.namedWindow(screen_name, cv2.WND_PROP_FULLSCREEN)

kinect = None
cap = None
    
"""
Grabs a depth map from the Kinect sensor and creates an image from it.
http://euanfreeman.co.uk/openkinect-python-and-opencv/
"""
def get_depth_map():    
    depth, timestamp = freenect.sync_get_depth()
    np.clip(depth, 0, 2**10 - 1, depth)
    return depth.astype(np.int)

def depth_map_to_bmp(depth):
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

WIDTH=640
HEIGHT=480
Y,X = np.meshgrid(np.arange(0,HEIGHT),np.arange(0,WIDTH),indexing='ij')
kinectXmat = X - (WIDTH/2.0)
kinectYmat = Y - (HEIGHT/2.0)
Min_Dist = -10.0
Scale_Factor = 0.0021

dmfloor,dmwall = np.meshgrid(np.arange(0,HEIGHT),np.arange(0,WIDTH),indexing='ij')
dmfloor = 1024 - 1024 * dmfloor / 480

def spiral(w,h):
    md = max(w,h)
    spiral = np.zeros((md,md),dtype=np.int32)
    # spiral = np.zeros((w,h),dtype=np.int32)
    sx = w/2 - 1
    sy = h/2 - 1
    cnt = 1
    spiral[sx,sy] = cnt
    maxcnt = md*md
    offset = 1
    llen = 2
    # directions of the spiral
    dirs = [(0,1),(-1,0),(0,-1),(1,0)]
    # start to the right
    sx += 1
    while(cnt < maxcnt):
        for dire in dirs:            
            for i in xrange(0,llen):
                sx += dire[0]
                sy += dire[1]
                print (sx,sy,cnt,llen,dire)
                spiral[sx,sy] = cnt
                cnt += 1
        sx +=1 # to the right
        llen += 2
    return spiral

spiral(4,4)

"""
	Conversion from http://openkinect.org/wiki/Imaging_Information
"""
def depth_map_to_points(dm):
    z = 0.1236 * np.tan(dm / 2842.5 + 1.1863)
    x = kinectXmat * (z + Min_Dist) * Scale_Factor
    y = kinectYmat * (z + Min_Dist) * Scale_Factor
    #x = (i - w / 2) * (z + minDistance) * scale_factor
    #y = (j - h / 2) * (z + minDistance) * scale_factor
    #z = z
    #Where
    #minDistance = -10
    #scaleFactor = .0021.
    return (x,y,z)

def points_to_depth_map(x,y,z):
    ''' returns z,x,y '''
    dm = 2842.5 * (np.arctan(z / 0.1236)  - 1.1863)
    ourX = (x / Scale_Factor) / (z + Min_Dist) 
    ourY = (y / Scale_Factor) / (z + Min_Dist) 
    return (dm,ourX,ourY)

def point_to_depth_map(x,y,z):
    return points_to_depth_map(x,y,z) # ha it works
    

def get_kinect_video():    
    if not kinect == None:
        return get_kinect_video_cv()
    depth, timestamp = freenect.sync_get_video()  
    if (depth == None):
        return None
    return depth[...,1]



def get_kinect_video_cv():    
    global kinect
    if kinect == None:
        print "Opening Kinect"
        kinect = cv2.VideoCapture(1)
    ret, frame2 = kinect.read()
    if not ret:
        return None
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    return next




def doFullscreen():
    global fullscreen
    if not fullscreen:
        cv2.setWindowProperty(screen_name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        fullscreen = True
    else:
        cv2.setWindowProperty(screen_name, cv2.WND_PROP_FULLSCREEN, 0)
        fullscreen = False

curr_state = None

handlers = dict()

def handle_keys():
    global fullscreen
    global handlers
    global curr_state
    k = cv2.waitKey(1000/60) & 0xff
    if k == 27:
        return True
    elif k == ord('f'):
        doFullscreen()
    else:
        if k in handlers:
            handlers[k]()
    return False


kernel = np.ones((5,5))





bounds = list()


def mix_point(c1, c2, prop=0.5):
    dprop = 1.0 - prop
    return (c1[0]*prop + c2[0]*dprop,
            c1[1]*prop + c2[1]*dprop,
            c1[2]*prop + c2[2]*dprop)

def min_max_point(minmax, newpoint):
    return (
        (
            max(minmax[0][0],newpoint[0]),
            max(minmax[0][1],newpoint[1]),
            max(minmax[0][2],newpoint[2])
        ),
        (
            min(minmax[1][0],newpoint[0]),
            min(minmax[1][1],newpoint[1]),
            min(minmax[1][2],newpoint[2])
        )
    )

def center_min_max(minmax):
    return ((minmax[0][0]+minmax[1][0])/2.0,
            (minmax[0][1]+minmax[1][1])/2.0,
            (minmax[0][2]+minmax[1][2])/2.0)


            
        
def communicate_centroid(centroid):
    send_osc("/kinect/centroid",*centroid)
    logging.info("Centroid sending (%s,%s,%s)" % centroid)

    
# step 1 get noise map

def get_mask(n=15):

    summap = get_depth_map().astype(np.int)
    summap *= 0
    n = 15
    for i in range(0,n):
        depth_map = get_depth_map().astype(np.int)
        depth_map = scipy.ndimage.morphology.grey_erosion(depth_map,size=(3,3))

        summap += (depth_map < 1) | (depth_map > 1020)
        logging.debug("%s %s" % (np.min(depth_map), np.max(depth_map)))
        logging.debug("%s %s %s" % (np.min(summap),np.max(summap),(np.sum(summap)/float(640*480))))

    mask = (summap == n) | (summap < (n*1/20)) * 1
    return (mask, summap)

context.mask, summap = get_mask(15)



my_timeout = 60.0


while(1):
    depth_map = get_depth_map()
    if depth_map == None:
        print "Bad?"
        continue
    
    # now communicate the centroid if there is motion
    
    depth_map_bmp = depth_map_to_bmp(depth_map)
    depth_map_bmp = cv2.flip(depth_map_bmp, 1)
    cv2.imshow(screen_name,(255/8) * (depth_map_bmp % 8))
    # cv2.imshow("%s - diff" % screen_name,depth_map_to_bmp(context.diff))# * context.diff))
    cv2.imshow("%s - diff" % screen_name,(context.diff == 0)*255.0)# * context.diff))

    if handle_keys():
        break


cap.release()
cv2.destroyAllWindows()
