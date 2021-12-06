'''   JLL, 2021.11.30
from /home/jinn/YPN/OPNet/modelB3c.py
Build modelB3 = UNet + Pose Net (PN)
pad Y_batch from 112 to 2383, path vector (pf5P) from 51 to 192 etc.
2383: see https://github.com/JinnAIGroup/OPNet/blob/main/output.txt
outs[0] = pf5P1 + pf5P2 = 385, outs[3] = rf5L1 + rf5L2 = 58
PWYbatch =  2383 - 2*192 - 1 - 2*29 = 1940

1. Use supercombo I/O
2. Task: Regression for Path Prediction
3. Input: 2 YUV images with 6 channels = (none, 12, 128, 256)
   #--- inputs.shape = (None, 12, 128, 256)
   #--- x0.shape = (None, 128, 256, 12)  # permutation layer
4. Output:
   #--- outputs.shape = (None, 2383)
Run:
  (YPN) jinn@Liu:~/YPN/OPNet$ python modelB3.py
'''
from tensorflow import keras
from tensorflow.keras import layers

def UNet(x0, num_classes):
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x0)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer (UNet final layer)
    x = layers.Conv2D(2*num_classes, 3, activation="softmax", padding="same")(x)

    # Add layers for PN
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)

    return x

# PN will be used in future
def PN(x):
    x1 = layers.Dense(64, activation='relu')(x)
    x2 = layers.Dense(64, activation='relu')(x)
    x3 = layers.Dense(64, activation='relu')(x)
    out1 = layers.Dense(385)(x1)
    out2 = layers.Dense(386)(x2)
    out3 = layers.Dense(386)(x3)
    out4 = layers.Dense(58)(x3)
    out5 = layers.Dense(200)(x3)
    out6 = layers.Dense(200)(x3)
    out7 = layers.Dense(200)(x3)
    out8 = layers.Dense(8)(x3)
    out9 = layers.Dense(4)(x3)
    out10 = layers.Dense(32)(x3)
    out11 = layers.Dense(524)(x3)
    #out12 = layers.Dense(512)(x3)
    outpts = layers.Concatenate(axis=-1)([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11])
    return (out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11)

def get_model(img_shape, num_classes):
    inputs = keras.Input(shape=img_shape)
    #--- inputs.shape = (None, 12, 128, 256)
    x0 = layers.Permute((2, 3, 1))(inputs)
    #--- x0.shape = (None, 128, 256, 12)
    x = UNet(x0, num_classes)
    out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11 = PN(x)

    # Define the model
    model = keras.Model(inputs, outputs=[out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11])
    #--- outputs.shape = (None, 2383)
    return model

if __name__=="__main__":
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    img_shape = (12, 128, 256)
    num_classes = 3
    model = get_model(img_shape, num_classes)
    model.summary()

    keras.utils.plot_model(model, "./saved_model/JL11_dlc_model.png")
    keras.utils.plot_model(model, "./saved_model/JL11_dlc_model.png", show_shapes=True) 



