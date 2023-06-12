import numpy as np
from train_utils import dice_coefficient
from typing import Tuple

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float]:
    """
    Evaluates accuracy and IoU (dice) scores for predicted masks.
    """
    accuracy = np.sum(y_true == y_pred)/y_true.shape[0]
    dice = dice_coefficient(y_true, y_pred)
    return accuracy, dice