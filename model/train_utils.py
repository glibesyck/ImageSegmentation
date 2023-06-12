from tensorflow import keras

CLASS_WEIGHTS = [0.01, 0.99]

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice metric implementation.
    """
    y_true = keras.backend.flatten(y_true)
    y_pred = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true * y_pred)
    union = keras.backend.sum(y_true) + keras.backend.sum(y_pred)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

def bce_loss(y_true, y_pred, smooth=1e-6):
    """
    Weighted binary cross-entropy loss implementation.
    """
    loss = keras.backend.mean(CLASS_WEIGHTS[1]*y_true*keras.backend.log(y_pred + smooth) + CLASS_WEIGHTS[0]*(1 - y_true) * keras.backend.log(1 - y_pred + smooth), axis=[1, 2, 3])
    return - loss

def combined_loss(y_true, y_pred, bce_weight=1):
    """
    Custom loss implementation taking into account 2 types of losses as well as imbalanced situation of dataset.
    """
    return bce_weight*bce_loss(y_true, y_pred) + (1 - bce_weight) * (1 - dice_coefficient(y_true, y_pred))