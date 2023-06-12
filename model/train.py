import wandb
import numpy as np
from tensorflow import keras
from train_utils import combined_loss, dice_coefficient

MODEL_PATH = "reduced_model.h5"

def train_model(model: keras.Model, train_ds: np.ndarray, val_ds: np.ndarray):
    opt = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=opt, loss=combined_loss, metrics=['accuracy', dice_coefficient])

    callbacks = [
        keras.callbacks.EarlyStopping("val_loss", patience = 15, verbose = True), 
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only = True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-7),
        wandb.keras.WandbCallback()
    ]

    model.fit(train_ds, validation_data = val_ds, epochs = 30, callbacks = callbacks)