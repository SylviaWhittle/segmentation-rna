from pathlib import Path
import re
from PIL import Image
from typing import Tuple

import numpy as np
import tensorflow as tf
from loguru import logger
from ruamel.yaml import YAML
from dvclive import Live
import matplotlib.pyplot as plt

from unet import dice_loss, iou_loss

yaml = YAML(typ="safe")


def dice(
    mask_predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes=[0, 1],
    epsilon=1e-6,
):
    dice_list = []
    for c in classes:
        y_true = ground_truth == c
        y_pred = mask_predicted == c
        intersection = 2.0 * np.sum(y_true * y_pred)
        dice_score = intersection / (np.sum(y_true) + np.sum(y_pred) + epsilon)
        dice_list.append(dice_score)
    return np.mean(dice_list)


def evaluate(
    model: tf.keras.Model,
    data_dir: Path,
    model_image_size: Tuple[int, int],
):
    """Evaluate the model on the test set.

    Parameters
    ----------
    model : tf.keras.Model
        The trained model.
    data_dir : Path
        The directory containing the test images and masks as separate numpy files. Eg. image_0.npy and mask_0.npy.
    """
    # Load the test images and masks
    logger.info("Evaluate: Loading the test images and masks.")

    # Find the indexes of all the image files in the format of image_<index>.npy
    image_indexes = [int(re.search(r"\d+", file.name).group()) for file in data_dir.glob("image_*.npy")]
    mask_indexes = [int(re.search(r"\d+", file.name).group()) for file in data_dir.glob("mask_*.npy")]

    # Check that the image and mask indexes are the same
    if set(image_indexes) != set(mask_indexes):
        raise ValueError(f"Different image and mask indexes : {image_indexes} and {mask_indexes}")

    with Live("results/evaluate") as live:
        dice_multi = 0.0

        for index in image_indexes:
            # Load the image and mask
            image = np.load(data_dir / f"image_{index}.npy")
            mask = np.load(data_dir / f"mask_{index}.npy")

            logger.info(f"Evaluate: Image index: {index}")
            logger.info(f"Evaluate: Image shape before reshape: {image.shape}")

            # Resize the image and mask to the model image size
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(model_image_size)
            image = np.array(pil_image)
            pil_mask = Image.fromarray(mask)
            pil_mask = pil_mask.resize(model_image_size)
            mask = np.array(pil_mask)

            logger.info(f"Evaluate: Image shape after reshape: {image.shape} | Mask shape: {mask.shape}")

            # Add the batch dimension
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

            logger.info(f"Evaluate: Image shape after adding batch dimension: {image.shape} | Mask shape: {mask.shape}")

            # Predict the mask
            mask_predicted = model.predict(image) > 0.5

            # Remove the batch dimension but keep the channel dimension as dice iterates over channels in case
            # of multi-class segmentation
            image = np.squeeze(image, axis=0)
            mask = np.squeeze(mask, axis=0)
            mask_predicted = np.squeeze(mask_predicted, axis=0)

            logger.info(
                f"Evaluate: Post-squeeze image shapes: Image: {image.shape} | Mask: {mask.shape} | Predicted Mask: {mask_predicted.shape}"
            )

            # Calculate the DICE score
            dice_score = dice(mask_predicted, mask)
            dice_multi += dice_score / len(image_indexes)

            # Plot the image, mask and predicted mask and log it
            num_channels = mask_predicted.shape[-1]
            fig, ax = plt.subplots(num_channels, 3, figsize=(15, 5))
            if num_channels == 1:
                ax[0].imshow(image[:, :, 0], cmap="viridis")
                ax[0].set_title("Image")
                ax[1].imshow(mask[:, :, 0], cmap="binary")
                ax[1].set_title("Ground Truth Mask")
                ax[2].imshow(mask_predicted[:, :, 0], cmap="binary")
                ax[2].set_title("Predicted Mask")
            else:
                for i in range(num_channels):
                    ax[i, 0].imshow(image[:, :, i], cmap="viridis")
                    ax[i, 0].set_title("Image")
                    ax[i, 1].imshow(mask[:, :, i], cmap="binary")
                    ax[i, 1].set_title(f"Ground Truth Mask Channel {i}")
                    ax[i, 2].imshow(mask_predicted[:, :, i], cmap="binary")
                    ax[i, 2].set_title(f"Predicted Mask Channel {i}")
            # plt.savefig(f"{plot_save_dir}/test_image_{index}.png")
            live.log_image(f"test_image_plot_{index}.png", fig)

        live.summary["dice_multi"] = dice_multi


if __name__ == "__main__":
    logger.info("Evaluate: Loading parameters from params.yaml config file.")
    # Get the parameters from the params.yaml config file
    with open(Path("./params.yaml"), "r") as file:
        all_params = yaml.load(file)
        base_params = all_params["base"]
        evaluate_params = all_params["evaluate"]

    logger.info("Evaluate: Converting the paths to Path objects.")
    # Convert the paths to Path objects
    model_path = Path(evaluate_params["model_path"])
    data_dir = Path(evaluate_params["test_data_dir"])

    # Get the right loss function and pass to custom objects
    custom_objects = {}
    loss_function = base_params["loss_function"]
    if loss_function == "dice_loss":
        custom_objects["dice_loss"] = dice_loss
    elif loss_function == "iou_loss":
        custom_objects["iou_loss"] = iou_loss
    elif loss_function == "binary_crossentropy":
        pass
    else:
        raise ValueError(f"Invalid loss function: {loss_function}")

    # Load the model
    logger.info("Evaluate: Loading the model.")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Evaluate the model
    logger.info("Evaluate: Evaluating the model.")
    evaluate(
        model=model,
        data_dir=data_dir,
        model_image_size=(base_params["model_image_size"], base_params["model_image_size"]),
    )
