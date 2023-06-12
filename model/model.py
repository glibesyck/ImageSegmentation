from tensorflow import keras

IMAGE_SIZE = (256, 256, 3)

def build_model(start_size = 4):
    inputs = keras.layers.Input((IMAGE_SIZE))

    #downsampling part

    conv11 = keras.layers.Conv2D(start_size, (3, 3), activation="relu", padding="same")(inputs)
    conv12 = keras.layers.Conv2D(start_size, (3, 3), activation="relu", padding="same")(conv11)
    max_pool1 = keras.layers.MaxPooling2D((2, 2))(conv12)
    drop1 = keras.layers.Dropout(0.2)(max_pool1)

    conv21 = keras.layers.Conv2D(start_size*2, (3, 3), activation="relu", padding="same")(drop1)
    conv22 = keras.layers.Conv2D(start_size*2, (3, 3), activation="relu", padding="same")(conv21)
    max_pool2 = keras.layers.MaxPooling2D((2, 2))(conv22)
    drop2 = keras.layers.Dropout(0.2)(max_pool2)

    conv31 = keras.layers.Conv2D(start_size*4, (3, 3), activation="relu", padding="same")(drop2)
    conv32 = keras.layers.Conv2D(start_size*4, (3, 3), activation="relu", padding="same")(conv31)
    max_pool3 = keras.layers.MaxPooling2D((2, 2))(conv32)
    drop3 = keras.layers.Dropout(0.2)(max_pool3)

    conv41 = keras.layers.Conv2D(start_size*8, (3, 3), activation="relu", padding="same")(drop3)
    conv42 = keras.layers.Conv2D(start_size*8, (3, 3), activation="relu", padding="same")(conv41)
    max_pool4 = keras.layers.MaxPooling2D((2, 2))(conv42)
    drop4 = keras.layers.Dropout(0.2)(max_pool4)

    #middle propagation part

    conv_mid1 = keras.layers.Conv2D(start_size*16, (3, 3), activation="relu", padding="same")(drop4)
    conv_mid2 = keras.layers.Conv2D(start_size*16, (3, 3), activation="relu", padding="same")(conv_mid1)
    drop5 = keras.layers.Dropout(0.2)(conv_mid2)

    #upsampling part

    deconv1 = keras.layers.Conv2DTranspose(start_size*8, (3, 3), strides=(2, 2), padding="same")(drop5)
    conc1 = keras.layers.concatenate([deconv1, conv42])
    conv51 = keras.layers.Conv2D(start_size*8, (3, 3), activation="relu", padding="same")(conc1)
    conv52 = keras.layers.Conv2D(start_size*8, (3, 3), activation="relu", padding="same")(conv51)
    drop6 = keras.layers.Dropout(0.2)(conv52)

    deconv2 = keras.layers.Conv2DTranspose(start_size*4, (3, 3), strides=(2, 2), padding="same")(drop6)
    conc2 = keras.layers.concatenate([deconv2, conv32])
    conv61 = keras.layers.Conv2D(start_size*4, (3, 3), activation="relu", padding="same")(conc2)
    conv62 = keras.layers.Conv2D(start_size*4, (3, 3), activation="relu", padding="same")(conv61)
    drop7 = keras.layers.Dropout(0.2)(conv62)

    deconv3 = keras.layers.Conv2DTranspose(start_size*2, (3, 3), strides=(2, 2), padding="same")(drop7)
    conc3 = keras.layers.concatenate([deconv3, conv22])
    conv71 = keras.layers.Conv2D(start_size*2, (3, 3), activation="relu", padding="same")(conc3)
    conv72 = keras.layers.Conv2D(start_size*2, (3, 3), activation="relu", padding="same")(conv71)
    drop8 = keras.layers.Dropout(0.2)(conv72)

    deconv4 = keras.layers.Conv2DTranspose(start_size, (3, 3), strides=(2, 2), padding="same")(drop8)
    conc4 = keras.layers.concatenate([deconv4, conv12])
    conv81 = keras.layers.Conv2D(start_size, (3, 3), activation="relu", padding="same")(conc4)
    conv82 = keras.layers.Conv2D(start_size, (3, 3), activation="relu", padding="same")(conv81)
    # drop9 = keras.layers.Dropout(0.2)(conv82)

    outputs = keras.layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(conv82)

    model = keras.Model(inputs = [inputs], outputs = [outputs])

    return model
