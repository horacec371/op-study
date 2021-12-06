"""   JLL, 2021.12.1
from /home/jinn/YPN/OPNet/datagenB3c.py
pad Ytrue from 112 to 2383, path vector (pf5P) from 51 to 192 etc.
2383: see https://github.com/JinnAIGroup/OPNet/blob/main/output.txt
outs[0] = pf5P1 + pf5P2 = 385, outs[3] = rf5L1 + rf5L2 = 58
PWYbatch =  2383 - 2*192 - 1 - 2*29 = 1940
make yuv.h5 by /home/jinn/openpilot/tools/lib/hevc2yuvh5B3.py

Input:
  bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=>  CbYUV (6, 291,  582) = (C, H, W) [key: 1311 =  874x3/2]
  sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=>  CsYUV (6, 128,  256) = (C, H, W) [key:  384 =  256x3/2]
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5
Output:
  Ximgs.shape = (none, 2x6, 128, 256)  (num_channels = 6, 2 yuv images)
  Ytrue1.shape = (none, 385), Ytrue2.shape = (none, 58), Ytrue3.shape = (none, 1940)
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def datagen(camera_files, batch_size):
    Ximgs = np.zeros((batch_size, 12, 128, 256), dtype='float32')   # for YUV imgs
    Ytrue1 = np.zeros((batch_size, 385), dtype='float32')
    Ytrue2 = np.zeros((batch_size, 386), dtype='float32')
    Ytrue3 = np.zeros((batch_size, 386), dtype='float32')
    Ytrue4 = np.zeros((batch_size, 58), dtype='float32')
    Ytrue5 = np.zeros((batch_size, 200), dtype='float32')
    Ytrue6 = np.zeros((batch_size, 200), dtype='float32')
    Ytrue7 = np.zeros((batch_size, 200), dtype='float32')
    Ytrue8 = np.zeros((batch_size, 8), dtype='float32')
    Ytrue9 = np.zeros((batch_size, 4), dtype='float32')
    Ytrue10 = np.zeros((batch_size, 32), dtype='float32')
    Ytrue11 = np.zeros((batch_size, 524), dtype='float32')
    #Ytrue12 = np.zeros((batch_size, 512), dtype='float32')    
    path_files  = [f.replace('yuv', 'pathdata') for f in camera_files]
    radar_files = [f.replace('yuv', 'radardata') for f in camera_files]
      #---  path_files  = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5']
      #---  radar_files = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5']
    for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
        if os.path.isfile(cfile) and os.path.isfile(pfile) and os.path.isfile(rfile):
            print('#---serverB3  OK: cfile, pfile, rfile exist')
        else:
            print('#---serverB3  Error: cfile, pfile, or rfile does not exist')
    batchIndx = 0
    Nplot = 0

    while True:
        for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
            with h5py.File(cfile, "r") as cf5:
                pf5 = h5py.File(pfile, 'r')
                rf5 = h5py.File(rfile, 'r')
                cf5X = cf5['X']
                pf5P = pf5['Path']
                rf5L = rf5['LeadOne']
                  #---  cf5['X'].shape       = (1150, 6, 128, 256)
                  #---  pf5['Path'].shape    = (1149, 51)
                  #---  rf5['LeadOne'].shape = (1149, 5)
                  #---  cf5X.shape           = (1150, 6, 128, 256)

                dataN = min(len(cf5X), len(pf5P), len(rf5L))
                PWPath   = 192 - pf5P.shape[1]   # PW = pad_width
                PWYll =  386
                PWYrl =  386
                PWLead   =  29 - rf5L.shape[1]
                PWYlx =  200
                PWYlv =  200
                PWYla =  200
                PWYds =  8
                PWYmeta =  4
                PWYdp =  32               
                PWYpose =  524
                #PWYspace =  512
                  #2383 - 2*192 - 1 - 2*29
                  #---  PWPath, PWLead, PWYbatch = 141 24 1940

                count = 0
                while count < batch_size:
                    ri = np.random.randint(0, dataN-2, 1)[-1]   # ri cannot be the last dataN-1
                    print("#---  count, dataN, ri =", count, dataN, ri)
                    for i in range(dataN-1):
                      if ri < dataN-1:
                        vsX1 = cf5X[ri]
                        vsX2 = cf5X[ri+1]
                          #---  vsX2.shape = (6, 128, 256)
                        Ximgs[count] = np.vstack((vsX1, vsX2))
                          #---  Ximgs[count].shape = (12, 128, 256)

                        pf5P1 = pf5P[ri]
                        pf5P2 = pf5P[ri+1]
                        rf5L1 = rf5L[ri]
                        rf5L2 = rf5L[ri+1]
                          #---  pf5P2.shape = (51,)
                          #---  rf5L2.shape = (5,)
                        pf5P1 = np.pad(pf5P1, (0, PWPath), 'constant')   # pad PWPath zeros ('constant') to the right; (0, PWPath) = (left, right)
                          #---  pf5P1.shape = (192,)
                        pf5P2 = np.pad(pf5P2, (0, PWPath+1), 'constant')   # +1 to 385
                        rf5L1 = np.pad(rf5L1, (0, PWLead), 'constant')
                        rf5L2 = np.pad(rf5L2, (0, PWLead), 'constant')
                        Y1 = np.hstack((pf5P1, pf5P2))
                        Y2 = []
                        Y2 = np.pad(Y2, (0, PWYll), 'constant')
                        Y3 = []
                        Y3 = np.pad(Y3, (0, PWYrl), 'constant')
                        Y4 = np.hstack((rf5L1, rf5L2))
                        Y5 = []
                        Y5 = np.pad(Y5, (0, PWYlx), 'constant')
                        Y6 = []
                        Y6 = np.pad(Y6, (0, PWYlv), 'constant')
                        Y7 = []
                        Y7 = np.pad(Y7, (0, PWYla), 'constant')
                        Y8 = []
                        Y8 = np.pad(Y8, (0, PWYds), 'constant')
                        Y9 = []
                        Y9 = np.pad(Y9, (0, PWYmeta), 'constant')
                        Y10 = []
                        Y10 = np.pad(Y10, (0, PWYdp), 'constant')
                        Y11 = []
                        Y11 = np.pad(Y11, (0, PWYpose), 'constant')
                        #Y12 = []
                        #Y12 = np.pad(Y12, (0, PWYspace), 'constant')
                        Ytrue1[count] = Y1
                        Ytrue2[count] = Y2
                        Ytrue3[count] = Y3
                        Ytrue4[count] = Y4
                        Ytrue5[count] = Y5
                        Ytrue6[count] = Y6
                        Ytrue7[count] = Y7
                        Ytrue8[count] = Y8
                        Ytrue9[count] = Y9
                        Ytrue10[count] = Y10
                        Ytrue11[count] = Y11
                        #Ytrue12[count] = Y12
                        
                        break
                    count += 1

                batchIndx += 1
                print('#---  batchIndx =', batchIndx)

                if Nplot == 0:
                    Yb = Ytrue1[0][:]
                    print('#---datagenB3  Ytrue1[0][50:51] =', Ytrue1[0][50:51])   # valid_len
                    print('#---datagenB3  Ytrue1[0][242:243] =', Ytrue1[0][242:243])   # valid_len, 242 = 50+192
                    print('#---datagenB3  Ytrue2[0][50:51] =', Ytrue2[0][0:1])   # lcar's d
                    print('#---datagenB3  Ytrue2[0][242:243] =', Ytrue2[0][29:30])   # lcar's d, 414 = 385+29
                    plt.plot(Yb)
                    plt.show()
                    Nplot += 1

                Ytrue1[:, 50:51]/=100   # normalize valid_len
                Ytrue1[:, 242:243]/=100
                Ytrue2[:, 0:1]/=100   # normalize lcar's d
                Ytrue2[:, 29:30]/=100

                if Nplot == 1:
                    Yb = Ytrue1[0][:]
                    print('#---datagenB3  Ytrue1[0][50:51] =', Ytrue1[0][50:51])   # valid_len
                    print('#---datagenB3  Ytrue1[0][242:243] =', Ytrue1[0][242:243])   # valid_len, 242 = 50+192
                    print('#---datagenB3  Ytrue2[0][50:51] =', Ytrue2[0][0:1])   # lcar's d
                    print('#---datagenB3  Ytrue2[0][242:243] =', Ytrue2[0][29:30])   # lcar's d, 414 = 385+29
                    plt.plot(Yb)
                    plt.show()
                    Nplot += 1

                  #---  Ximgs.shape = (16, 12, 128, 256)
                  #---  Ytrue.shape = (16, 2383)
                yield Ximgs, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11
                
                
                
                
                
                
                
                
