from tensorflow import keras
import numpy as np
from train_utils import combined_loss, dice_coefficient
from dataset import ShipDetection

def predict_model(model_path: str, test_ds: ShipDetection) -> np.ndarray:
    """
    Returns predicted segmentation masks for given test dataset.
    """
    model = keras.models.load_model(model_path, custom_objects={'combined_loss': combined_loss, 'dice_coefficient': dice_coefficient})
    predictions = model.predict(test_ds)
    predictions = np.where(predictions > 0.5, 1.0, 0.0)
    return predictions

