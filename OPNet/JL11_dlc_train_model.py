"""   JLL, 2021.12.1
from /home/jinn/YPN/OPNet/train_modelB3c.py
train modelB3 = UNet + Pose Net (PN)
pad Y_batch from 112 to 2383, path vector (pf5P) from 51 to 192 etc.
2383: see https://github.com/JinnAIGroup/OPNet/blob/main/output.txt
outs[0] = pf5P1 + pf5P2 = 385, outs[3] = rf5L1 + rf5L2 = 58
PWYbatch =  2383 - 2*192 - 1 - 2*29 = 1940

1. Use OP supercombo I/O
2. Task: Regression for Path Prediction
3. Pose: 56 = Ego path 51 (x, y) + 5 radar lead car's dRel (relative distance), yRel, vRel (velocity), aRel (acceleration), prob
4. Ground Truth: Y_batch = (none, 2383)
5. Loss: mean squared error (mse)
6. Input:
   X_batch from yuv.h: 2 YUV images with 6 channels = (none, 12, 128, 256)
   Y_batch from pathdata.h5, radardata.h5
7. Output: mse loss
8. Make yuv.h5 by /home/jinn/openpilot/tools/lib/hevc2yuvh5B3.py

Run: on 3 terminals
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverB3.py --port 5557
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverB3.py --port 5558 --validation
   (YPN) jinn@Liu:~/YPN/OPNet$ python train_modelB3.py --port 5557 --port_val 5558
Input:
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5, pathdata.h5, radardata.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--33/yuv.h5, pathdata.h5, radardata.h5
Output:
  /OPNet/saved_model/modelB3_loss.npy
  
Training History:
  BATCH_SIZE = 16  EPOCHS = 2
  loss: 0.0177 - val_loss: 0.0154
  Training Time: 00:02:44.61
"""
import os
import h5py
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from JL11_dlc_model import get_model
from JL11_dlc_server import client_generator, BATCH_SIZE

EPOCHS = 2

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def gen(hwm, host, port, model):
    for tup in client_generator(hwm=hwm, host=host, port=port):
        Ximgs, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11 = tup

          #---  Ximgs.shape = (16, 12, 128, 256)
          #---  Ytrue.shape = (16, 2383)
        yield Ximgs, (Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11)

  # multiple losses: https://stackoverflow.com/questions/53707199/keras-combining-two-losses-with-adjustable-weights-where-the-outputs-do-not-have
  # https://stackoverflow.com/questions/58230074/how-to-optimize-multiple-loss-functions-separately-in-keras
def custom_loss(y_true, y_pred):
      #print('#---  y_true[0, :] =', y_true[0, :])
      #print('#---  y_pred[0, :] =', y_pred[0, :])
    loss1 = tf.keras.losses.mse(y_true[0], y_pred[0])
    loss2 = tf.keras.losses.mse(y_true[1], y_pred[1])
    loss3 = tf.keras.losses.mse(y_true[2], y_pred[2])
    loss4 = tf.keras.losses.mse(y_true[3], y_pred[3])  
    loss5 = tf.keras.losses.mse(y_true[4], y_pred[4])
    loss6 = tf.keras.losses.mse(y_true[5], y_pred[5])
    loss7 = tf.keras.losses.mse(y_true[6], y_pred[6])
    loss8 = tf.keras.losses.mse(y_true[7], y_pred[7])
    loss9 = tf.keras.losses.mse(y_true[8], y_pred[8])
    loss10 = tf.keras.losses.mse(y_true[9], y_pred[9])
    loss11 = tf.keras.losses.mse(y_true[10], y_pred[10])      
    loss = 0.5*loss1 + 0.5*loss2

    return loss

if __name__=="__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='Training modelB3')
    parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    parser.add_argument('--port', type=int, default=5557, help='Port of server.')
    parser.add_argument('--port_val', type=int, default=5558, help='Port of server for validation dataset.')
    args = parser.parse_args()

    # Build model
    img_shape = (12, 128, 256)
    num_classes = 6
    model = get_model(img_shape, num_classes)
    #model.summary()

    filepath = "./saved_model/JL11_dlc_model-BestWeights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    #model.load_weights('./saved_model/modelB3-BestWeights.hdf5', by_name=True)
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=custom_loss)

    history = model.fit(
        gen(20, args.host, port=args.port, model=model),
        steps_per_epoch=1150//BATCH_SIZE, epochs=EPOCHS,
        validation_data=gen(20, args.host, port=args.port_val, model=model),
        validation_steps=1150//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
      # steps_per_epoch=1150//BATCH_SIZE = 71  use this for longer training

    model.save('./saved_model/JL11_dlc_model.h5')

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    np.save('./saved_model/JL11_dlc_model_loss', np.array(history.history['loss']))
    lossnpy = np.load('./saved_model/JL11_dlc_model_loss.npy')
    plt.plot(lossnpy)  #--- lossnpy.shape = (10,)
    plt.draw() #plt.show()
    plt.pause(0.5)
    input("Press ENTER to exit ...")
    plt.close()
