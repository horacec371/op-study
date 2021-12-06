''' Horace Chen, 2021.09.15; JLL, 2021.11.19
1. Correct Leon's errors in visualizer.py (HC)
2. Find frame rate and speed up matplotlib.pyplot animation (JL)
3. Correct  imgs_med_model[0] = imgs_med_model[1] (JL)
4. Update for the preliminary PC-based simualtor. 2021.12.06 
(YPN) jinn@Liu:~/YPN/Leon$ python simulator101.py ./fcamera.hevc
Error:
  /home/jinn/YPN/Leon/common/lanes_image_space.py:89: RuntimeWarning: divide by zero encountered in double_scalars
  p_image = p_full_frame = np.array([KEp[0] / KEp[2], KEp[1] / KEp[2], 1.])
'''
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.lanes_image_space import transform_points
from common.tools.lib.parser import parser

camerafile = sys.argv[1]
supercombo = load_model('models/JL11_dlc_model.h5', compile = False)
#print(supercombo.summary())

def frames_to_tensor(frames):
  H = (frames.shape[1]*2)//3
  W = frames.shape[2]
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

  in_img1[:, 0] = frames[:, 0:H:2, 0::2]
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

imgs_med_model = np.zeros((2, 384, 512), dtype=np.uint8)
state = np.zeros((1,512))
desire = np.zeros((1,8))

cap = cv2.VideoCapture(camerafile)
fps = cap.get(cv2.CAP_PROP_FPS)

x_left = x_right = x_path = np.linspace(0, 192, 192)

(ret, previous_frame) = cap.read()
  #print ("#--- frome no. = ", cap.get(cv2.CAP_PROP_POS_FRAMES))
  #--- frome no. =  1.0
  #cap.release()

if not ret:
   exit()
else:
  frame_no = 1
  img_yuv = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YUV_I420)
  imgs_med_model[0] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
    #--- imgs_med_model.shape = (2, 384, 512)
fig = plt.figure('OPNet Simulator 101')
#plt.subplots_adjust( left=0.1, right=1.5, top=1.5, bottom=0.1, wspace=0.2, hspace=0.2)

while True:
  (ret, current_frame) = cap.read()
  if not ret:
       break
  frame_no += 1
  #print ("#--- frame_no =", frame_no)

  frame = current_frame.copy()
  img_yuv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV_I420)
  imgs_med_model[1] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))

  if frame_no > 0:
  
    plt.clf()
    plt.title("lanes and path")
    plt.xlim(0, 1200)
    plt.ylim(800, 0)

    frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0
      #--- frame_tensors.shape = (2, 6, 128, 256)
    inputs = [np.vstack(frame_tensors[0:2])[None], desire, state]
    outs = supercombo.predict(inputs)
    parsed = parser(outs)
      # Important to refeed the state
    state = outs[-1]
    pose = outs[-2]   # For 6 DoF Callibration
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.clf()
    ##################
    plt.subplot(221)
    plt.title("Transformed, Merged, and Plotted")
    new_x_left, new_y_left = transform_points(x_left, parsed["lll"][0])
    new_x_right, new_y_right = transform_points(x_left, parsed["rll"][0])
    new_x_path, new_y_path = transform_points(x_left, parsed["path"][0])

    plt.plot(new_x_left, new_y_left, label='transformed', color='b')
    plt.plot(new_x_right, new_y_right, label='transformed', color='r')
    plt.plot(new_x_path, new_y_path, label='transformed', color='g')

    plt.imshow(frame) # HC: Merge raw image and plot together

    ##################
    plt.subplot(222)
    plt.gca().invert_yaxis()
    plt.title("Improved Plot")
    new_x_left, new_y_left = transform_points(x_left, parsed["lll"][0])
    new_x_right, new_y_right = transform_points(x_left, parsed["rll"][0])
    new_x_path, new_y_path = transform_points(x_left, parsed["path"][0])

    plt.plot(new_x_left, new_y_left, label='transformed', color='b')
    plt.plot(new_x_right, new_y_right, label='transformed', color='r')
    plt.plot(new_x_path, new_y_path, label='transformed', color='g')
    
    plt.legend(['left', 'right', 'path'])
    
    ##################
    plt.subplot(223) # Resize image
    frame = cv2.resize(frame, (640, 420))
    plt.title("Resize to 640X420")
    plt.imshow(frame)	     

    ##################
    plt.subplot(224)
    plt.gca().invert_xaxis()
    plt.title("Inverted X axis: Lanes and Path")
      # From main.py
      # lll = left lane line
    plt.plot(parsed["lll"][0], range(0,192), "b-", linewidth=1)
      # rll = right lane line
    plt.plot(parsed["rll"][0], range(0, 192), "r-", linewidth=1)
      # path = path cool isn't it ?
    plt.plot(parsed["path"][0], range(0, 192), "g-", linewidth=1)
    
    #plt.legend(['lll', 'rll', 'path'])
      # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis

    if frame_no < 100: 
        print(frame_no,': ','new_x_path = ',new_x_path[:3],'parsed["path"][0] = ', parsed["path"][0][:3]) # check update. 2021.12.06
    	
    plt.pause(0.001)
    if cv2.waitKey(10) & 0xFF == ord('q'):
          break

  imgs_med_model[0] = imgs_med_model[1]

print ("#--- Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
