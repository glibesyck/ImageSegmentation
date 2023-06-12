import numpy as np 
import pandas as pd
import os
from PIL import Image
from typing import Tuple, List
import keras

START_IMAGE_SIZE = (768, 768, 3)
END_IMAGE_SIZE = (256, 256, 3)
MASK_SIZE = (256, 256, 1)

def RLE_to_img(mask: pd.DataFrame, image_id: str) -> np.ndarray:
    """
    Additional function to return segmentation mask in numpy array format.
    """
    segmentation = mask[mask["ImageId"] == image_id]["EncodedPixels"].values[0]
    image = np.zeros(START_IMAGE_SIZE[0] * START_IMAGE_SIZE[1], dtype="float32")
    if not segmentation:
        return image.reshape(START_IMAGE_SIZE[0], START_IMAGE_SIZE[1])
    segmentation = segmentation.strip().split(" ")   

    start_pixels = np.array([(int(x) - 1) for x in segmentation[::2]]) #to start from 0, not 1
    lengths = np.array([int(x) for x in segmentation[1::2]])
    end_pixels = start_pixels + lengths

    for index in range(start_pixels.shape[0]):
        image[start_pixels[index]:end_pixels[index]] = 1
    image = image.reshape(START_IMAGE_SIZE[0], START_IMAGE_SIZE[1]).T

    return image

class ShipDetection(keras.utils.Sequence):
    """
    Dataset class with images and their corresponding segmentation masks.
    """
    def __init__(self, image_ids: List[int], mask_path: str, image_folder: str, batch_size: int):
        self.mask = pd.read_csv(mask_path)
        self.image_ids = image_ids
        self.image_folder = image_folder
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_ids) // self.batch_size + 1

    def __getitem__(self, index):
        batch_ids = self.image_ids[index*self.batch_size:min((index+1)*self.batch_size, len(self.image_ids))]

        inputs = np.zeros((min(self.batch_size, len(self.image_ids) - index*self.batch_size),) + END_IMAGE_SIZE, dtype="float32")
        outputs = np.zeros((min(self.batch_size, len(self.image_ids) - index*self.batch_size),) + MASK_SIZE, dtype="float32")
 
        for j, image_id in enumerate(batch_ids):
            mask = RLE_to_img(self.mask, image_id)
            mask = np.asarray(Image.fromarray(mask).resize((MASK_SIZE[0], MASK_SIZE[1])), dtype="float32").reshape(MASK_SIZE)
            outputs[j] = mask
            img = np.asarray(Image.open(os.path.join(self.image_folder, image_id)).resize((END_IMAGE_SIZE[0], END_IMAGE_SIZE[1])), dtype="float32").reshape(END_IMAGE_SIZE)
            inputs[j] = img / 255  # normalizing

        return inputs, outputs

def create_dataset(train_ids: List[int], val_ids: List[int], test_ids: List[int], mask_path: str, image_folder: str) -> Tuple[ShipDetection, ShipDetection, ShipDetection]:
    train_generator = ShipDetection(train_ids, mask_path, image_folder, batch_size=32)
    val_generator = ShipDetection(val_ids, mask_path, image_folder, batch_size=32)
    test_generator = ShipDetection(test_ids, mask_path, image_folder, batch_size=32)

    return train_generator, val_generator, test_generator







