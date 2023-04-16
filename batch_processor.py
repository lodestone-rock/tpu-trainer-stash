import concurrent.futures as cft
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd
from transformers import CLIPTokenizer


def process_image(
    image_path: str,
    rescale_size: Sequence[int],
) -> np.array:
    r"""
    scale the image resolution to predetermined resolution and return
    it as numpy

    args:
        image_path (:obj:`str`):
            path to file
        rescale_size (:obj:`list` or `tuple`):
            width and height target
    return: np.array
    """
    image = np.flip(cv2.imread(image_path, cv2.IMREAD_COLOR), axis=-1)
    dimen = np.array(image.shape[:2])
    axis_keys = np.argsort(dimen)
    minor_axis, major_axis = axis_keys
    # rescale image with scaling factor
    scale_factor = np.max(rescale_size / dimen)
    scale = tuple(np.rint(dimen * scale_factor).astype(int))[::-1]
    inter = cv2.INTER_LINEAR
    image = cv2.resize(image, scale, interpolation=inter)
    # get smallest and largest res from image
    # warning
    if dimen[major_axis] < max(rescale_size):
        print(
            f"[WARN] image {image_path} is smaller than designated batch, zero pad will be added"
        )

    delta = (image.shape[major_axis] - image.shape[minor_axis]) // 2
    if delta != 0:
        index = [slice(None)] * image.ndim
        index[major_axis] = slice(delta, -delta)
        image = image[tuple(index)]
    # cheeky resize to catch missmatch
    if image.shape[:2] != rescale_size:
        image = cv2.resize(image, rescale_size, interpolation=inter)
    # normalize
    image = image / 127.5 - 1
    # HWC -> CHW
    image = image.swapaxes(-3, -1)
    return image


def tokenize_text(
    tokenizer: CLIPTokenizer,
    text_prompt: list,
    max_length: int,
    batch_slice: int = 1,
) -> dict:
    r"""
    wraps huggingface tokenizer function with some batching functionality
    convert long token for example (1,1002) to (1,10,102)
    start and end token are extracted and reappended for each batch

    args:
        tokenizer (:obj:`CLIPTokenizer`):
            tokenizer class
        text_prompt (:obj:`list`):
            batch text to be tokenized
        max_length (:obj:`int`):
            maximum token before clipping
        batch_slice (:obj:`int`, *optional*, defaults to 1):
            if greater than 1 it will slice the token into batch evenly
            (max_length-2) must be divisible by this value

    return:
        dict:
            {"attention_mask": np.array, "input_ids": np.array}
    """

    # check
    assert (
        max_length - 2
    ) % batch_slice == 0, "(max_length-2) must be divisible by batch_slice"

    text_input = tokenizer(
        text=text_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )

    max_length = tokenizer.model_max_length
    if batch_slice > 1:
        # ###[stack input ids]### #
        value = text_input["input_ids"]
        # strip start and end token
        # [start, token1, token2, ..., end] to
        # [token1, token2, ..., tokenN]
        content = value[:, 1:-1].reshape(-1, batch_slice, max_length - 2)
        # store start and end token and then reshape it to be concatenated
        start = np.full(
            shape=(content.shape[0], content.shape[1], 1), fill_value=[value[:, 0][0]]
        )
        stop = np.full(
            shape=(content.shape[0], content.shape[1], 1), fill_value=[value[:, -1][0]]
        )
        # concat start and end token
        # from shape (batch, 75*3+2)
        # to shape (batch, 3, 77)
        new_value = np.concatenate([start, content, stop], axis=-1)
        text_input["input_ids"] = new_value

        # ###[stack attention mask]### #
        mask = text_input["attention_mask"]
        # strip start and end mask
        # [start, mask1, mask2, ..., end] to
        # [mask1, mask2, ..., maskN]
        content = mask[:, 1:-1].reshape(-1, batch_slice, max_length - 2)
        # store start and end mask and then reshape it to be concatenated
        start = np.full(
            shape=(content.shape[0], content.shape[1], 1), fill_value=[mask[:, 0][0]]
        )
        # concat start and end mask
        # from shape (batch, 75*3+2)
        # to shape (batch, 3, 77)
        new_value = np.concatenate([start, start, content], axis=-1)
        text_input["attention_mask"] = new_value

    return text_input


def generate_batch(
    process_image_fn: Callable[[str, tuple], np.array],
    tokenize_text_fn: Callable[[str, str, int], dict],
    tokenizer: CLIPTokenizer,
    dataframe: pd.DataFrame,
    image_name_col: str,
    score_col: str,
    caption_col: str,
    caption_token_length: int,
    width_col: str,
    height_col: str,
    batch_slice: int = 1,
    executor: cft.Executor = None,
) -> dict:
    """
    generate a single batch for training.
    use this function in a for loop while swapping the dataframe batch
    depends on process_image and tokenize_text function

    args:
        process_image_fn (:obj:`Callable`):
            process_image function
        process_image_fn (:obj:`Callable`):
            tokenize_text function
        tokenizer (:obj:`CLIPTokenizer`):
            tokenizer class
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        folder_path (:obj:`str`):
            path to image folder
        image_name_col (:obj:`str`):
            column name inside dataframe filled with image names
        caption_col (:obj:`str`):
            column name inside dataframe filled with text captions
        caption_token_length (:obj:`int`):
            maximum token before clipping
        tokenizer_path (:obj:`str`):
            path to file / hugging face path
        width_col (:obj:`str`):
            column name inside dataframe filled with bucket width of an image
        height_col (:obj:`str`):
            column name inside dataframe filled with bucket height of an image
        batch_slice (:obj:`int`, *optional*, defaults to 1):
            if greater than 1 it will slice the token into batch evenly
            (caption_token_length-2) must be divisible by this value
    return:
        dict:
            {
                "attention_mask": np.array,
                "input_ids": np.array,
                "pixel_values": np.array
            }
    """
    # count batch size
    batch_image = []

    # ###[process image]### #
    # process batch sequentialy
    image_names = dataframe[image_name_col]
    image_sizes = zip(dataframe[width_col], dataframe[height_col])
    mapper = map if executor is None else executor.map
    batch_image = list(mapper(process_image_fn, image_names, image_sizes))
    # stack image into neat array
    batch_image = np.stack(batch_image)
    # as contiguous array
    batch_image = np.ascontiguousarray(batch_image)

    # ###[process token]### #
    batch_prompt = dataframe.loc[:, caption_col].tolist()
    tokenizer_dict = tokenize_text_fn(
        tokenizer=tokenizer,
        text_prompt=batch_prompt,
        max_length=caption_token_length,
        batch_slice=batch_slice,
    )
    image_score = dataframe[score_col].values
    image_score = np.expand_dims(image_score, axis=-1)
    output = {
        "pixel_values": batch_image,
        "image_scores": image_score,
        **tokenizer_dict,
    }
    return output
