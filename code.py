import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx
from random import randint
from matplotlib import pyplot as plt

mouse = Controller()
app = wx.App(False)
(sx,sy) = wx.GetDisplaySize()
(camx,camy) = (240,320)
cam = cv2.VideoCapture(0)
cam.set(3,camx)
cam.set(4,camy)
mLocOld = np.array([0,0])
mouseLoc =  np.array([0,0])
df = 2
tile_s = 60
S_s_d = 58000
over = 5


def create_mask(imgHSV,lower_bound,upper_bound,kernelOpen,kernelClose,eraser = 0):
    h = imgHSV.shape[0]
    mask = cv2.inRange(imgHSV,lower_bound,upper_bound)
    maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    if eraser == True:
        for x in range(h):
            if x<64 or x>219:
                maskClose[x] = np.zeros(maskClose[x].shape[0])
    return maskClose

def mask_on_original(img,mask):
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_img = cv2.addWeighted(img,1,mask,1,0)
    return masked_img

def texture_synthesis(img,tile_size,overlap):
    '''
    1. create a patch
    2. Calculate SSD,
        if SSD is good, blend image
        else create a new patch
    '''
    ssd_threshold = S_s_d
    tile = create_block(img,tile_size,0,0)
    h,w = img.shape[0],img.shape[1]
    for u in range(h):
        for v in range(w):
            if img[u][v][0] == 255 and img[u][v][1] == 255 and img[u][v][2] == 255:
                num_white_pixels += 1
                SSD = ssd_threshold
                count = 0
                while(SSD >= ssd_threshold):
                    count +=1
                    SSD,io,to = calculate_SSD(img,tile,u,v,overlap)
                    if SSD < ssd_threshold:
                        img = blend_img(img,tile,u,v,io,to)
                    else:
                        tile = create_block(img,tile_size,u,v)
                        if count >30:
                           img = blend_img(img,tile,u,v,io,to)
                           SSD = 1
    return img 
    
def create_block(img,tile_size,u,v):
    tile = np.zeros((tile_size, tile_size, 3), np.uint8)
    h,w = img.shape[0],img.shape[1]
    count_white = [1]
    x = random_xy(u-tile_size,11,u+tile_size,h-tile_size-20)
    y = random_xy(v-tile_size,15,v+tile_size,w-tile_size-100)
    while len(count_white)>0:
        for i in range(tile_size):
            for j in range(tile_size):
                tile[i][j] = img[x+i][y+j]
        count_white,_,_ = np.where(tile == (255, 255, 255))
        x = random_xy(u-tile_size,11,u+tile_size,h-tile_size-20)
        y = random_xy(v-tile_size,15,v+tile_size,w-tile_size-100)
    return tile

def random_xy(a,b,c,d):
    return randint(min(max(a,b),c,d),max(a,b,min(c,d)))

def calculate_SSD(img,tile,u,v,overlap):
    tile_size_x,tile_size_y = tile.shape[0],tile.shape[1]
    ssd_top = 0
    ssd_left = 0
    ssd_down = 0
    ssd_right = 0
    diff= np.zeros((3))
    x = tile_size_x-overlap
    y = tile_size_y-overlap
    img_overlap = {}
    tile_overlap = {}
    try:
        for i in range(tile_size_x):
            for j in range(overlap):
                diff[0] = (img[u+i][v-(overlap-j)][0] - tile[i][j][0])
                diff[1] = (img[u+i][v-(overlap-j)][1] - tile[i][j][1])
                diff[2] = (img[u+i][v-(overlap-j)][2] - tile[i][j][2])
                ssd_left += (diff[0]**2 + diff[1]* 2 + diff[2]**2)**0.5
        img_overlap['l'] = [u,v-overlap,tile_size_x,overlap]
        tile_overlap['l'] =  [0,0,tile_size_x,overlap]
    except:
        ssd_left = ssd_left
    
    try:
        for i in range(overlap):
            for j in range(tile_size_y):
                diff[0] = (img[u-(overlap-i)][v+j][0] - tile[i][j][0])
                diff[1] = (img[u-(overlap-i)][v+j][1] - tile[i][j][1])
                diff[2] = (img[u-(overlap-i)][v+j][2] - tile[i][j][2])
                ssd_top += (diff[0]**2 + diff[1]* 2 + diff[2]**2)**0.5
        img_overlap['t'] = [u-overlap,v,overlap,tile_size_y]
        tile_overlap['t'] =  [0,0,overlap,tile_size_y]
    except:
        ssd_top = ssd_top

    try:
        if img[u][v+y][0]!=255 and img[u][v+y][1]!=255 and img[u][v+y][2]!=255:
            for i in range(tile_size_x):
                for j in range(overlap):
                    diff[0] = (img[u+i][v+j+y][0] - tile[i][y+j][0])
                    diff[1] = (img[u+i][v+j+y][1] - tile[i][y+j][1])
                    diff[2] = (img[u+i][v+j+y][2] - tile[i][y+j][2])
                    ssd_right += (diff[0]**2 + diff[1]* 2 + diff[2]**2)**0.5
            img_overlap['r'] = [u,v+y,tile_size_x,overlap]
            tile_overlap['r'] =  [0,y,tile_size_x,overlap]
    except:
        ssd_right = ssd_right
    try:
        if img[u+x][v][0]!=255 and img[u+x][v][1]!=255 and img[u+x][v][2]!=255:
            for i in range(overlap):
                for j in range(tile_size_y):
                    diff[0] = (img[u+x+i][v+j][0] - tile[x+i][j][0])
                    diff[1] = (img[u+x+i][v+j][1] - tile[x+i][j][1])
                    diff[2] = (img[u+x+i][v+j][2] - tile[x+i][j][2])
                    ssd_down += (diff[0]**2 + diff[1]* 2 + diff[2]**2)**0.5
            
            img_overlap['d'] = [u+x,v,overlap,tile_size_y]
            tile_overlap['d'] =  [x,0,overlap,tile_size_y]
    except:
        ssd_down = ssd_down

    return max(ssd_top,ssd_left,ssd_right,ssd_down), img_overlap,tile_overlap

def chunk(i_t,overlap):
    d = [] 
    u,v,h,w = overlap[0],overlap[1],overlap[2],overlap[3]
    for x in range(u,u+h):
        inter = []
        for y in range(v,v+w):            
            inter.append(i_t[x][y])
        d.append(inter)
    return np.array(d)

def blend_img(img,tile,u,v,io,to):
    paste_tile = True
    for key in io:
        image_overlap = np.array(chunk(img,io[key]))
        tile_overlap = np.array(chunk(tile,to[key]))
        tile_size_x,tile_size_y = tile.shape[0],tile.shape[1]
        blended_img = 0.2 * image_overlap + 0.8 * tile_overlap
        i,j,h,w = io[key][0],io[key][1],io[key][2],io[key][3]
        s_x,s_y = min (u,i), min(v,j)    
        for x in range(tile_size_x):
            for y in range(tile_size_y):
                if s_x+x in range(i,i+h-1) and s_y+y in range(j,j+w-1):
                    img[s_x+x][s_y+y]= blended_img[s_x+x-i][s_y+y-j]
                else:
                    if paste_tile:
                        img[s_x+x][s_y+y] = tile[x][y]
                        paste_tile = False
    return img

vid_path = "Original.avi"
vid= cv2.VideoCapture(vid_path)
vid.set(3,320)
vid.set(4,240)
e_img = vid.read()[1]
HSVimg = cv2.cvtColor(e_img, cv2.COLOR_BGR2HSV)
mask = create_mask(HSVimg,np.array([0,0,0]),np.array([25,255,255]),np.ones((3,3), np.uint8),np.ones((3,3), np.uint8),1)
masked_img = mask_on_original(e_img,mask)
synthesized_img = texture_synthesis(masked_img,tile_s,over)

# mouse callback function
def smart_eraser(event,x,y,flags,param):
    if(len(conts)==1):
        for i in range(-5,5):
            for j in range(-5,5):
                e_img[y+i][x+j] = synthesized_img[y+i][x+j]

cv2.namedWindow('image')
cv2.setMouseCallback('image',smart_eraser)

while(1):
    ret, img = cam.read()
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cursor_mask = create_mask(imgHSV,np.array([33,80,40]),np.array([102,255,255]),np.ones((3,3)),np.ones((10,10)))
    conts, hier = cv2.findContours(cursor_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if(len(conts)==1):
        x,y,w,h = cv2.boundingRect(conts[0])
        cv2.rectangle(img,(x-2,y-2),(x+w+2,y+h+2),(255,0,0),2)
        cx = int(x+w/2)
        cy = int(y+h/2)
        cv2.circle(img,(cx,cy),2,(0,255,0),2)
        mouseLoc = mLocOld + ((cx,cy)-mLocOld)/df
        mouse.position=(sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy)
        mLocOld = mouseLoc
    cv2.imshow('video_CAM',img)
    cv2.imshow('image',e_img)
    if 27==cv2.waitKey(10) & 0XFF:
        break
cv2.destroyAllWindows()