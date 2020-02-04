# FixMatch for ImageNet

This directory contains implementation of FixMatch for ImageNet with RandAugment and CT Augment.

## Prerequisites

1. Follow [these instructions](./prep_imagenet_data/README.md) to prepare splits of semi-supervised data for ImageNet training.

2. Make sure to have python 3, pip and virual env installed.

    ```
    sudo apt install python3-dev python3-virtualenv
    ```

3. Install all software dependecies needed for Tensorflow 2.1.

  * If you plan to run code on GPUs you have to install NVIDIA driver, CUDA and cuDNNs,
    see [Tensorflow documentation](https://www.tensorflow.org/install/gpu).
    Keep in mind which versions of these libraries works with Tensorflow 2.1.

3. Create python virtual environment and install all necessary dependencies:

   ```
   virtualenv -p python3 ~/.venv3/fixmatch_imagenet
   source ~/.venv3/fixmatch_imagenet/bin/activate
   pip install tensorflow==2.1
   pip install tensorflow-addons
   pip install absl-py
   pip install easydict
   ```

   * It is recommended to avoid `--system-site-packages`, especially if you already have
     Tensorflow installed on the system. Otherwise you may get binary incopatibility
     between Tensorflow and Tensorflow Addons, see https://github.com/tensorflow/addons/issues/676#issuecomment-560159918 for context.

4. Make sure that `${SSL_IMAGENET_DIR}` points to directory with semi-supervised
   ImageNet training data.

## Training ImageNet models

Provided codebase supports training on GPU and Cloud TPU.

### Training on GPU

Train fully supervised model with CT Augmentation:

```
mkdir -p ${HOME}/models/supervised
python -B supervised.py \
  --imagenet_data="${SSL_IMAGENET_DIR}" \
  --model_dir="${HOME}/models/supervised" \
  --steps_per_run=100 \
  --dataset="imagenet" \
  --hparams="\"bfloat16\": false, \"num_epochs\": 200, \"augment\": { \"type\": \"cta\" }"
```

Train fixmatch model on 10% ImageNet data with random augmentation (random magnitude):

```
mkdir -p ${HOME}/models/fixmatch
python -B fixmatch.py \
  --imagenet_data="${SSL_IMAGENET_DIR}" \
  --model_dir="${HOME}/models/fixmatch" \
  --steps_per_run=100 \
  --per_worker_batch_size=8 \
  --dataset="imagenet128116.1" \
  --hparams="\"bfloat16\": false, \"num_epochs\": 3000, \"learning_rate\": { \"warmup_epochs\": 50, \"decay_epochs\": 500 }, \"augment\": { \"type\": \"randaugment\" }"
```

### Training on Cloud TPU

Running TPU training is very similar to running GPU training with following exceptions:

* You have to create Cloud TPU instance and provide address to this instance to training code via `--tpu` argument. See aslo Cloud TPU [documentation](https://cloud.google.com/tpu/docs/creating-deleting-tpus).
* Both dataset and model have to be stored on Google Cloud Storage. See [Cloud TPU resnet example](https://cloud.google.com/tpu/docs/tutorials/resnet-2.x) for possible setup.
* You can use `bfloat16` data type when training on TPU by specifying it in the hyperparameters list. This usually results in faster training with similar accuracy.

Example of running fixmatch model on TPU:

```
python -B fixmatch.py \
  --imagenet_data="${SSL_IMAGENET_DIR}" \
  --tpu="${CLOUD_TPU_INSTANCE}" \
  --model_dir="${MODEL_DIR}" \
  --steps_per_run=1000 \
  --per_worker_batch_size=32 \
  --dataset="imagenet128116.1" \
  --hparams="\"bfloat16\": true, \"num_epochs\": 3000, \"learning_rate\": { \"warmup_epochs\": 50, \"decay_epochs\": 500 }, \"augment\": { \"type\": \"randaugment\" }"
```


### Training parameters

Training is controlled by following command line arguments:

* `--steps_per_run` controls how many training steps are done between evaluations.
  It also controls how often CT augmentation parameters are updates, when CTA is used.
  1000 steps is recommended for TPU trainings.
* `--per_worker_batch_size` controls per worker (GPU or TPU core) supervised batch size,
  with default 128. Note that unsupervised batch size is computed by multiplying `per_worker_batch_size` by `uratio` hyperparameter.
* `--hparams` is a list of hyperparameters in JSON format.
  `DEFAULT_COMMON_HPARAMS` in `training.py` and `DEFAULT_FIXMATCH_HPARAMS` in `fixmatch.py`
  contains default values of all hyperparameters.
  * Note that number of epochs (`num_epochs` hyperparameter) is measures
    in epochs of supervised examples.
    So each epoch on 10% of ImageNet data is 10 times shorter compared to full ImageNet data.
* `--dataset` controls dataset used for training.
  It has to be `imagenet` for supervised training. For semi-supervised training on 10% of
  ImageNet data it should be `imagenet128116.${split}` where `${split}` is a split of
  semisupervised data, which could be one of `1`, `2`, `3`, `4` or `5`.