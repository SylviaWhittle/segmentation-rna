from pathlib import Path
from typing import Tuple
from PIL import Image
from loguru import logger
from datetime import datetime
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from ruamel.yaml import YAML

from unet import unet_model

yaml = YAML(typ="safe")


# generator for data
def image_data_generator(
    data_dir: Path,
    image_indexes: np.ndarray,
    batch_size: int,
    model_image_size: Tuple[int, int],
    norm_upper_bound: float,
    norm_lower_bound: float,
):
    """Generate batches of images and ground truth masks."""

    while True:
        # Select files for the batch
        batch_indexes = np.random.choice(a=image_indexes, size=batch_size, replace=False)
        batch_input = []
        batch_output = []

        # Load images and ground truth
        for index in batch_indexes:
            # Load the image and ground truth
            image = np.load(data_dir / f"image_{index}.npy")
            ground_truth = np.load(data_dir / f"mask_{index}.npy").astype(bool)

            # TODO: Augment the images: Scale and translate

            # Resize without interpolation
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(model_image_size, resample=Image.NEAREST)
            image = np.array(pil_image)

            pil_ground_truth = Image.fromarray(ground_truth)
            pil_ground_truth = pil_ground_truth.resize(model_image_size, resample=Image.NEAREST)
            ground_truth = np.array(pil_ground_truth)

            # Normalise the image
            image = np.clip(image, norm_lower_bound, norm_upper_bound)
            image = image - norm_lower_bound
            image = image / (norm_upper_bound - norm_lower_bound)

            # TODO: Augment images: Flipping and rotate

            # Add the image and ground truth to the batch
            batch_input.append(image)
            batch_output.append(ground_truth)

        # Convert the lists to numpy arrays
        batch_x = np.array(batch_input).astype(np.float32)
        batch_y = np.array(batch_output).astype(np.float32)

        # logger.info(f"Batch x shape: {batch_x.shape}")
        # logger.info(f"Batch y shape: {batch_y.shape}")

        yield (batch_x, batch_y)


def train_model(
    random_seed: int,
    train_data_dir: Path,
    model_save_dir: Path,
    model_image_size: Tuple[int, int],
    activation_function: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    norm_upper_bound: float,
    norm_lower_bound: int,
    validation_split: float,
    loss_function: str,
):
    """Train a model to segment images."""

    logger.info("Training: Setup")

    logger.info("Training: Parameters:")
    logger.info(f"|  Random seed: {random_seed}")
    logger.info(f"|  Train data directory: {train_data_dir}")
    logger.info(f"|  Model save directory: {model_save_dir}")
    logger.info(f"|  Model image size: {model_image_size}")
    logger.info(f"|  Activation function: {activation_function}")
    logger.info(f"|  Learning rate: {learning_rate}")
    logger.info(f"|  Batch size: {batch_size}")
    logger.info(f"|  Epochs: {epochs}")
    logger.info(f"|  Normalisation upper bound: {norm_upper_bound}")
    logger.info(f"|  Normalisation lower bound: {norm_lower_bound}")
    logger.info(f"|  Test size: {validation_split}")
    logger.info(f"|  Loss function: {loss_function}")

    # Set the random seed
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    logger.info("Training: Loading data")
    # Find the indexes of all the image files in the format of image_<index>.npy
    image_indexes = [int(re.search(r"\d+", file.name).group()) for file in train_data_dir.glob("image_*.npy")]
    mask_indexes = [int(re.search(r"\d+", file.name).group()) for file in train_data_dir.glob("mask_*.npy")]

    # Check that the image and mask indexes are the same
    if set(image_indexes) != set(mask_indexes):
        raise ValueError(f"Different image and mask indexes : {image_indexes} and {mask_indexes}")

    # Train test split
    train_indexes, validation_indexes = train_test_split(
        image_indexes, test_size=validation_split, random_state=random_seed
    )
    logger.info(f"Training on {len(train_indexes)} images | validating on {len(validation_indexes)} images.")

    # Create an image data generator
    logger.info("Training: Creating data generators")

    train_generator = image_data_generator(
        data_dir=train_data_dir,
        image_indexes=train_indexes,
        batch_size=batch_size,
        model_image_size=model_image_size,
        norm_upper_bound=norm_upper_bound,
        norm_lower_bound=norm_lower_bound,
    )

    validation_generator = image_data_generator(
        data_dir=train_data_dir,
        image_indexes=validation_indexes,
        batch_size=batch_size,
        model_image_size=model_image_size,
        norm_upper_bound=norm_upper_bound,
        norm_lower_bound=norm_lower_bound,
    )

    # Load the model
    logger.info("Training: Loading model")
    model = unet_model(
        image_height=model_image_size[0],
        image_width=model_image_size[1],
        image_channels=1,
        learning_rate=learning_rate,
        activation_function=activation_function,
        loss_function=loss_function,
    )

    steps_per_epoch = len(train_indexes) // batch_size
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # At the end of each epoch, DVCLive will log the metrics
    logger.info("Using DVCLive to log the metrics.")
    with Live("results/train") as live:

        logger.info("Training the model.")
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch,
            verbose=1,
            callbacks=[DVCLiveCallback(live=live)],
        )

        logger.info("Training: Finished training.")

        logger.info(f"Training: Saving model to {model_save_dir}")
        model.save(Path(model_save_dir) / "catsnet_model.keras")
        live.log_artifact(
            str(Path(model_save_dir) / "catsnet_model.keras"),
            type="model",
            name="catsnet_model",
            desc="Model trained to segment cats.",
            labels=["cv", "segmentation"],
        )
        logger.info("Training: Finished.")

        # loss = history.history["loss"]
        # val_loss = history.history["val_loss"]
        # epoch_indexes = range(1, len(loss) + 1)
        # plt.plot(epoch_indexes, loss, "bo", label="Training loss")
        # plt.plot(epoch_indexes, val_loss, "b", label="Validation loss")
        # plt.title("Training and validation loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.show()

        # date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Save the model
        # model.save(model_save_dir / f"model_{date}.h5")


if __name__ == "__main__":
    logger.info("Train: Loading the parameters from the params.yaml config file.")
    # Get the parameters from the params.yaml config file
    with open(Path("./params.yaml"), "r") as file:
        all_params = yaml.load(file)
        base_params = all_params["base"]
        train_params = all_params["train"]

    logger.info("Train: Converting the paths to Path objects.")
    # Convert the paths to Path objects
    train_data_dir = Path(train_params["train_data_dir"])
    model_save_dir = Path(train_params["model_save_dir"])

    # Train the model
    train_model(
        random_seed=base_params["random_seed"],
        train_data_dir=train_data_dir,
        model_save_dir=model_save_dir,
        model_image_size=(base_params["model_image_size"], base_params["model_image_size"]),
        activation_function=train_params["activation_function"],
        learning_rate=train_params["learning_rate"],
        batch_size=train_params["batch_size"],
        epochs=train_params["epochs"],
        norm_upper_bound=train_params["norm_upper_bound"],
        norm_lower_bound=train_params["norm_lower_bound"],
        validation_split=train_params["validation_split"],
        loss_function=base_params["loss_function"],
    )
