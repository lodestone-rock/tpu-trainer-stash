# python stuff
import os

os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = "1"  # memory defrag do not disable!
import concurrent.futures as cft
import logging
import random
import sys
import time
from multiprocessing import Process, Queue

import jax
import jax.numpy as jnp

# basic stuff
import pandas as pd

# store cache xla compilation so you don't
# have to wait everything to compile again ever
from jax.experimental.compilation_cache import compilation_cache as cc
from tqdm.auto import tqdm

cache_dir = "/tmp/jax_reusable_cache"
if jax.devices()[0].platform == "tpu":
    cc.initialize_cache(cache_dir)

import optax
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard

# all ML stuff
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel

# local import
from batch_processor import generate_batch, process_image, tokenize_text
from dataframe_processor import (
    discrete_scale_to_equal_area,
    resolution_bucketing_batch_with_chunking,
    scale_by_minimum_axis,
    tag_suffler_to_comma_separated,
)

start_epoch = 0
number_of_epoch = 10


def main(epoch=0, steps_offset=0, lr=2e-6):
    # ===============[global var]=============== #

    # master seed
    seed = 69 + epoch  # noice

    # pandas bucketing
    csv_file = "posts.csv"
    image_dir = "e6_dump/resized"
    batch_num = 8
    batch_size = 8 * batch_num
    maximum_resolution_area = [512**2]
    bucket_lower_bound_resolution = [512]
    maximum_axis = 1024
    minimum_axis = 512
    # if true maximum_resolution_area and bucket_lower_bound_resolution not used
    # else maximum_axis and minimum_axis is not used
    use_ragged_batching = False
    repeat_batch = 10

    # batch generator (dataloader)
    image_name_col = "file"
    width_height = ["new_image_width", "new_image_height"]
    caption_col = "newtag_string"
    token_concatenate_count = 3
    token_length = 75 * token_concatenate_count + 2
    score_col = "fav_count"
    use_sam = True

    # diffusers model
    # initial model
    base_model_name = "size-512-squared_no-eos-bos_shuffled_lion-optim_custom-loss-e"
    model_dir = f"e6_dump/{base_model_name}{epoch}"  # continue from last model
    weight_dtype = jnp.bfloat16  # mixed precision training
    optimizer_algorithm = "lion"
    adam_to_lion_scale_factor = 7
    text_encoder_learning_rate = lr / 4 * batch_num
    u_net_learning_rate = lr * batch_num
    text_encoder_learning_rate = text_encoder_learning_rate
    save_step = 6000
    # saved model name
    model_name = f"{base_model_name}{epoch+1}"
    output_dir = f"e6_dump/{model_name}"
    print_loss = True
    debug = False  # enable to perform short training loop
    average_loss_step_count = 100
    # let unet decide the color to not be centered around 0 mean
    # enable only at the last epoch
    use_offset_noise = False
    strip_bos_eos_token = True

    # logger
    log_file_output = "logs.txt"
    loss_csv = f"{model_name}.csv"

    # ===============[logger]=============== #

    logging.basicConfig(
        # filename=log_file_output,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_output),
            logging.StreamHandler(sys.stdout),
        ],
        level=logging.INFO,
    )

    with open(loss_csv, "w") as loss_file:
        loss_file.write(f"global_step,loss,time")

    logging.info("init logs")
    logging.info(f"model dir: {model_dir}")

    # ===============[Initialize training RNG]=============== #

    rng = jax.random.PRNGKey(seed)
    train_rngs = jax.random.split(rng, jax.local_device_count())
    logging.info("generate RNG")

    # ===============[pandas batching & bucketing]=============== #
    # ensure image exist
    data = pd.read_csv(csv_file)
    rng = jax.random.split(rng)[1]
    image_list = os.listdir(image_dir)
    data = data.loc[data[image_name_col].isin(image_list)]
    data = data.set_index("md5").join(
        pd.read_csv("e6score/posts.csv", usecols=["md5", score_col], index_col="md5")
    )
    mask = data[score_col].isna()
    data[score_col][mask] = 0.0
    # create bucket resolution
    if use_ragged_batching:
        data_processed = scale_by_minimum_axis(
            dataframe=data,
            image_height_col="image_height",
            image_width_col="image_width",
            new_image_height_col="new_image_height",
            new_image_width_col="new_image_width",
            target_minimum_scale=minimum_axis,
            target_maximum_scale=maximum_axis,
        )

    else:
        # check guard
        assert len(maximum_resolution_area) == len(
            bucket_lower_bound_resolution
        ), "list count not match!"
        # multiple aspect ratio training!
        image_properties = zip(maximum_resolution_area, bucket_lower_bound_resolution)
        store_multiple_aspect_ratio = []

        for aspect_ratio in image_properties:
            data_processed = discrete_scale_to_equal_area(
                dataframe=data,
                image_height_col_name="image_height",
                image_width_col_name="image_width",
                new_image_height_col_name="new_image_height",
                new_image_width_col_name="new_image_width",
                max_res_area=aspect_ratio[0],
                bucket_lower_bound_res=aspect_ratio[1],
                extreme_aspect_ratio_clip=2.0,
                aspect_ratio_clamping=2.0,
                return_with_helper_columns=False,
            )
            store_multiple_aspect_ratio.append(data_processed)

        data_processed = pd.concat(store_multiple_aspect_ratio)

    # generate bucket batch and provide starting batch
    # with all possible image resolution to make sure jax compile everything in one go
    data_processed = resolution_bucketing_batch_with_chunking(
        dataframe=data_processed,
        image_height_col_name="new_image_height",
        image_width_col_name="new_image_width",
        seed=seed,
        bucket_batch_size=batch_size,
        repeat_batch=repeat_batch,
        bucket_group_col_name="bucket_group",
    )

    # shuffle tags
    def shuffle(tags, seed):
        tags = tags.split(",")
        random.Random(len(tags) * seed).shuffle(tags)
        tags = ",".join(tags)
        return tags

    data_processed[caption_col] = data_processed[caption_col].apply(
        lambda x: shuffle(x, seed)
    )
    logging.info("creating bucket and dataloader sequence")

    # ===============[load model to CPU]=============== #

    ckpt_name = (
        model_dir
        if os.path.exists(model_dir)
        else "lodestones/stable-diffusion-v1-5-flax"
    )
    tokenizer = CLIPTokenizer.from_pretrained(ckpt_name, subfolder="tokenizer")

    text_encoder = FlaxCLIPTextModel.from_pretrained(
        ckpt_name, subfolder="text_encoder", dtype=weight_dtype
    )

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        ckpt_name,
        dtype=weight_dtype,
        subfolder="vae",
    )

    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        ckpt_name, subfolder="unet", dtype=weight_dtype, use_memory_efficient=True
    )

    logging.info("load models to TPU")

    # ===============[optimizer function]=============== #

    if optimizer_algorithm == "adamw":
        # optimizer for U-Net
        u_net_constant_scheduler = optax.constant_schedule(u_net_learning_rate)
        u_net_adamw = optax.adamw(
            learning_rate=u_net_constant_scheduler,
            b1=0.9,
            b2=0.999,
            eps=1e-08,
            weight_decay=1e-2,
        )
        u_net_optimizer = optax.chain(
            optax.clip_by_global_norm(1),  # prevent explosion
            u_net_adamw,
        )

        # optimizer for CLIP text encoder
        text_encoder_constant_scheduler = optax.constant_schedule(
            text_encoder_learning_rate
        )
        text_encoder_adamw = optax.adamw(
            learning_rate=text_encoder_constant_scheduler,
            b1=0.9,
            b2=0.999,
            eps=1e-08,
            weight_decay=1e-2,
        )
        text_encoder_optimizer = optax.chain(
            optax.clip_by_global_norm(1),  # prevent explosion
            text_encoder_adamw,
        )

    if optimizer_algorithm == "lion":
        u_net_constant_scheduler = optax.constant_schedule(
            u_net_learning_rate / adam_to_lion_scale_factor
        )
        text_encoder_constant_scheduler = optax.constant_schedule(
            text_encoder_learning_rate / adam_to_lion_scale_factor
        )

        # optimizer for U-Net
        u_net_lion = optax.lion(
            learning_rate=u_net_constant_scheduler,
            b1=0.9,
            b2=0.99,
            weight_decay=1e-2 * adam_to_lion_scale_factor,
        )
        u_net_optimizer = optax.chain(
            optax.clip_by_global_norm(1),  # prevent explosion
            u_net_lion,
        )

        # optimizer for CLIP text encoder
        text_encoder_lion = optax.lion(
            learning_rate=text_encoder_constant_scheduler,
            b1=0.9,
            b2=0.99,
            weight_decay=1e-2 * adam_to_lion_scale_factor,
        )
        text_encoder_optimizer = optax.chain(
            optax.clip_by_global_norm(1),  # prevent explosion
            text_encoder_lion,
        )

    logging.info(f"setup optimizer: {optimizer_algorithm}")

    # ===============[train state and scheduler]=============== #

    unet_state = train_state.TrainState.create(
        apply_fn=unet.__call__, params=unet_params, tx=u_net_optimizer
    )

    text_encoder_state = train_state.TrainState.create(
        apply_fn=text_encoder.__call__,
        params=text_encoder.params,
        tx=text_encoder_optimizer,
    )

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    logging.info("create U-Net and CLIP text encoder train state")

    # ===============[replicate model parameters and state on each device]=============== #

    unet_state = jax_utils.replicate(unet_state)
    text_encoder_state = jax_utils.replicate(text_encoder_state)
    vae_params = jax_utils.replicate(vae_params)
    noise_scheduler_state = noise_scheduler.create_state()
    logging.info("replicate model weights and biases to each TPU")

    # ===============[train function]=============== #

    def train_step(unet_state, text_encoder_state, vae_params, batch, train_rng):
        # generate rng and return new_train_rng to be used for the next iteration step
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, num=3)

        params = {"text_encoder": text_encoder_state.params, "unet": unet_state.params}

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params},
                batch["pixel_values"],
                deterministic=True,
                method=vae.encode,
            )

            # get sample distribution from VAE latent
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            # weird scaling don't touch it's a lazy normalization
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            # I think I should combine this with the first noise seed generator
            noise_offset_rng, noise_rng, timestep_rng = jax.random.split(
                sample_rng, num=3
            )
            noise = jax.random.normal(noise_rng, latents.shape)
            if use_offset_noise:
                # mean offset noise, why add offset?
                # here https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise_offset = (
                    jax.random.normal(
                        noise_offset_rng, (latents.shape[0], latents.shape[1], 1, 1)
                    )
                    * 0.1
                )
                noise = noise + noise_offset

            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(
                noise_scheduler_state, latents, noise, timesteps
            )
            print(batch["input_ids"].shape)
            # batch["input_ids"] shape (batch, token_append, token)
            batch_dim = batch["input_ids"].shape[0]
            token_append_dim = batch["input_ids"].shape[1]

            # reshape batch["input_ids"] to shape (batch & token_append, token)
            input_ids = batch["input_ids"].reshape(-1, batch["input_ids"].shape[-1])
            # Get the text embedding for conditioning
            # encoder_hidden_states shape (batch & token_append, token, hidden_states)
            encoder_hidden_states = text_encoder_state.apply_fn(
                input_ids,
                params=params["text_encoder"],
                dropout_rng=dropout_rng,
                train=True,
            )[0]
            print(encoder_hidden_states.shape)
            # reshape encoder_hidden_states to shape (batch, token_append, token, hidden_states)
            encoder_hidden_states = encoder_hidden_states.reshape(
                (batch_dim, token_append_dim, -1, encoder_hidden_states.shape[-1])
            )
            print(encoder_hidden_states.shape)

            if strip_bos_eos_token:
                encoder_hidden_states = jnp.concatenate(
                    [
                        # first encoder hidden states without eos token
                        encoder_hidden_states[:, 0, :-1, :],
                        # the rest of encoder hidden states without both bos and eos token
                        jnp.reshape(
                            encoder_hidden_states[:, 1:-1, 1:-1, :],
                            (
                                encoder_hidden_states.shape[0],
                                -1,
                                encoder_hidden_states.shape[-1],
                            ),
                        ),
                        # last encoder hidden states without bos token
                        encoder_hidden_states[:, -1, 1:, :],
                    ],
                    axis=1,
                )
            else:
                # reshape encoder_hidden_states to shape (batch, token_append & token, hidden_states)
                encoder_hidden_states = jnp.reshape(
                    encoder_hidden_states,
                    (
                        encoder_hidden_states.shape[0],
                        -1,
                        encoder_hidden_states.shape[-1],
                    ),
                )
            print(encoder_hidden_states.shape)

            # Predict the noise residual because predicting image is hard :P
            # essentially try to undo the noise process
            model_pred = unet.apply(
                {"params": params["unet"]},
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                train=True,
            ).sample

            # Get the target for loss depending on the prediction type
            # sd1.x use epsilon aka noise residual but sd2.1 use velocity prediction
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    noise_scheduler_state, latents, noise, timesteps
                )
            else:
                # panic!!
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            l2_err = optax.l2_loss(target, model_pred).mean(
                axis=tuple(range(model_pred.ndim)[1:])
            )
            labels = batch["image_scores"]
            # Try to regress more heavily onto images with high scores than low scores
            l2_err, labels = map(
                lambda a: jax.lax.all_gather(a, axis="batch", tiled=True),
                (l2_err, labels),
            )
            density = jax.nn.softmax(labels)

            def dos_loss(l2_err, labels, density, lmbda_=0.25):
                """
                Try to regress more heavily onto images with high scores than
                low scores. Weight by inverse of return density.
                https://arxiv.org/abs/2301.12842
                """
                sorter = labels.reshape(-1).argsort()[::-1]
                density = jnp.take(density.reshape(-1), sorter).reshape(-1, 1)
                weights = jnp.triu(density @ density.T, k=1)
                l2_err = jnp.take(l2_err.reshape(-1), sorter).reshape(-1, 1)
                scores = -l2_err - jnp.logaddexp(-l2_err, -lmbda_ * l2_err.T)
                invmag = 1 / jnp.sum(weights)
                return -jnp.sum(weights * scores * invmag)

            return dos_loss(l2_err, labels, density)

        # perform autograd
        if use_sam:
            grad = jax.grad(compute_loss)(params)
            ascent_stride = 0.01 / optax.global_norm(grad)
            descent_params = optax.apply_updates(
                params, jax.tree_util.tree_map(lambda dw: dw * ascent_stride, grad)
            )
        else:
            descent_params = params
        loss, grad = jax.value_and_grad(compute_loss)(descent_params)
        # update weight and bias value
        new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
        new_text_encoder_state = text_encoder_state.apply_gradients(
            grads=grad["text_encoder"]
        )

        # calculate loss
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_unet_state, new_text_encoder_state, metrics, new_train_rng

    logging.info("define train step function")

    # ===============[compile to device]=============== #

    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1))

    logging.info("jit pmap train step function")

    # ===============[save model]=============== #

    def checkpoint(unet_state, text_encoder_state, vae_params, output_dir):
        # get the first of 8 replicated weights and biases to be saved
        def get_params_to_save(params):
            return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))

        # save using different scheduler because this one is prefered for inference
        scheduler, _ = FlaxPNDMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )
        # Create the pipeline using the trained modules and save it.
        pipeline = FlaxStableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            ),
        )

        # save it
        pipeline.save_pretrained(
            output_dir,
            params={
                "text_encoder": get_params_to_save(text_encoder_state.params),
                "vae": get_params_to_save(vae_params),
                "unet": get_params_to_save(unet_state.params),
            },
        )

    # ===============[simple dataloader]=============== #

    # spawn dataloader in another core
    def generate_batch_wrapper(list_of_batch: list, queue: Queue):
        # loop until queue is full
        with cft.ThreadPoolExecutor() as thread_pool:
            for batch in list_of_batch:
                current_batch = generate_batch(
                    process_image_fn=process_image,
                    tokenize_text_fn=tokenize_text,
                    tokenizer=tokenizer,
                    dataframe=data_processed.iloc[
                        batch * batch_size : batch * batch_size + batch_size
                    ],
                    image_name_col=image_name_col,
                    caption_col=caption_col,
                    caption_token_length=token_length,
                    width_col=width_height[0],
                    height_col=width_height[1],
                    batch_slice=token_concatenate_count,
                    score_col=score_col,
                    executor=thread_pool,
                )
                # put task in queue
                queue.put(current_batch)

    # ===============[training loop]=============== #

    logging.info("start training")

    # get group index as batch order
    assert (
        len(data_processed) % batch_size == 0
    ), f"DATA IS NOT CLEANLY DIVISIBLE BY {batch_size} {len(data_processed)%batch_size}"
    batch_order = list(range(0, len(data_processed) // batch_size))

    batch_order = batch_order[steps_offset:]

    # perfom short training run for debugging purposes
    if debug:
        batch_order = batch_order[:1000]
        save_step = 100
        average_loss_step_count = 20

    training_step = 0

    train_step_progress_bar = tqdm(
        total=len(batch_order), desc="Training...", position=1, leave=False
    )

    # loop counter
    train_metric = None
    sum_train_metric = 0
    global_step = 0

    # store training array here
    batch_queue = Queue(maxsize=10)

    # spawn another process for processing images
    batch_processor = Process(
        target=generate_batch_wrapper, args=[batch_order, batch_queue, debug]
    )
    batch_processor.start()
    start = time.time()

    for x in batch_order:
        # grab training array from queue
        current_batch = batch_queue.get()

        # (current_batch)
        # split it to multiple devices
        batch = shard(current_batch)

        # update loading bar
        train_step_progress_bar.update(1)

        # save periodically
        if global_step % save_step == 0:
            # save model and try to save at the begining
            for attempt in range(10):
                try:
                    checkpoint(
                        unet_state,
                        text_encoder_state,
                        vae_params,
                        output_dir,
                    )
                    logging.info(
                        f"=======================[saving models at {global_step} step(s)]======================="
                    )
                except Exception as e:
                    time.sleep(2)
                    print(e)
                    continue
                else:
                    break
            pass
        # this line of code block jax dispatch if enabled
        if print_loss:
            if train_metric != None:
                # accumulate loss value to be averaged
                loss = train_metric["loss"][0]
                sum_train_metric = sum_train_metric + loss

                # calculate average loss
                if global_step % average_loss_step_count == 0:
                    loss = sum_train_metric / average_loss_step_count
                    stop = time.time()
                    time_elapsed = stop - start
                    train_step_progress_bar.write(
                        f"Training... Loss:{loss} took {time_elapsed} second(s)"
                    )
                    start = time.time()
                    # save loss to csv
                    with open(loss_csv, "a") as loss_file:
                        loss_file.write(f"\n{global_step},{loss},{time_elapsed}")
                    # reset sum
                    sum_train_metric = 0

        # train model!!
        # this function run asynchronously and will continue without
        # blocking the loop, so all function above will get executed multiple
        # times until internal dispatch queue is full unless blocked by function
        # please check this train_rngs, i have concern if this train_rngs got dispatched
        # with the same value. technically it shouldn't tho
        unet_state, text_encoder_state, train_metric, train_rngs = p_train_step(
            unet_state, text_encoder_state, vae_params, batch, train_rngs
        )

        # increment train step
        global_step = global_step + 1

    train_step_progress_bar.close()
    # save model
    for attempt in range(10):
        try:
            checkpoint(unet_state, text_encoder_state, vae_params, output_dir)
        except Exception as e:
            time.sleep(2)
            print(e)
            continue
        else:
            break
    batch_processor.join()
    logging.info(f"=======================[finished training]=======================")
    # TODO: save model[done], TQDM[done], logger[done], loss[done], batch counter[done], wandb?


# epoch loop
for epoch in range(start_epoch, number_of_epoch):
    # offset of current epoch (useful when there's outage)
    steps_offset = [0] * len(range(start_epoch, number_of_epoch))
    # start from batch number X
    # useful when resuming training in the middle of the epoch
    steps_offset[0] = 0

    main(epoch=epoch, steps_offset=0)
