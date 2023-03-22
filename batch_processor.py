from typing import Union, Callable
import PIL
from PIL import ImageFile, Image
import pandas as pd
import numpy as np
import pathlib
from transformers import CLIPTokenizer


ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_image(
    image_path: str,
    rescale_size: Union[list, tuple],
    upper_bound: int = 10,
    debug: bool = False,
) -> Union[np.array, tuple]:
    r"""
    scale the image resolution to predetermined resolution and return
    it as numpy

    args:
        image_path (:obj:`str`):
            path to file
        rescale_size (:obj:`list` or `tuple`):
            width and height target
        upper_bound (:obj:`int`, *optional*, defaults to 10):
            major axis obund (not important, just set it as high as possible)
        debug (:obj:`bool`, *optional*, defaults to `False`):
            will return tuple (np.array, PIL.Image)

    return: np.array or (np.array, PIL.Image)
    """
    image = Image.open(image_path)

    # find the scaling factor for each axis
    x_scale = rescale_size[0] / image.size[0]
    y_scale = rescale_size[1] / image.size[1]
    scaling_factor = max(x_scale, y_scale)

    # rescale image with scaling factor
    new_scale = [
        round(image.size[0] * scaling_factor),
        round(image.size[1] * scaling_factor),
    ]
    sampling_algo = PIL.Image.NEAREST
    image = image.resize(new_scale, resample=sampling_algo)

    # get smallest and largest res from image
    minor_axis_value = min(image.size)
    minor_axis = image.size.index(minor_axis_value)
    major_axis_value = max(image.size)
    major_axis = image.size.index(major_axis_value)

    # warning
    if max(image.size) < max(rescale_size):
        print(
            f"[WARN] image {image_path} is smaller than designated batch, zero pad will be added"
        )

    if minor_axis == 0:
        # left and right same crop top and bottom
        top = (image.size[1] - rescale_size[1]) // 2
        bottom = (image.size[1] + rescale_size[1]) // 2

        # remainder add
        bottom_remainder = top + bottom
        # left, top, right, bottom
        image = image.crop((0, top, image.size[0], bottom))
    else:
        # top and bottom same crop the left and right
        left = (image.size[0] - rescale_size[0]) // 2
        right = (image.size[0] + rescale_size[0]) // 2
        # left, top, right, bottom
        image = image.crop((left, 0, right, image.size[1]))

    # cheeky resize to catch missmatch
    image = image.resize(rescale_size, resample=sampling_algo)
    # for some reason np flip width and height
    np_image = np.array(image)
    # normalize
    np_image = np_image / 127.5 - 1
    # height width channel to channel height weight
    np_image = np.transpose(np_image, (2, 0, 1))
    # add batch axis
    # np_image = np.expand_dims(np_image, axis=0)

    if debug:
        return (np_image, image)
    else:
        return np_image


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
    folder_path: str,
    image_name_col: str,
    score_col: str,
    caption_col: str,
    caption_token_length: int,
    tokenizer_path: str,
    width_col: str,
    height_col: str,
    batch_slice: int = 1,
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
    batch_size = len(dataframe)
    batch_image = []

    # ###[process image]### #
    # process batch sequentialy
    for x in range(batch_size):
        # get image name and size from datadrame
        image_name = dataframe.iloc[x][image_name_col]
        width_height = [dataframe.iloc[x][width_col], dataframe.iloc[x][height_col]]

        # grab iamge from path and then process it
        image_path = pathlib.Path(folder_path, image_name)
        image = process_image_fn(image_path, width_height)

        batch_image.append(image)
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
