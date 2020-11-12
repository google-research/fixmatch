# FixMatch

Code for the paper: "[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)" by 
Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel.

This is not an officially supported Google product.

![FixMatch diagram](media/FixMatch%20diagram.png)

## Setup

**Important**: `ML_DATA` is a shell environment variable that should point to the location where the datasets are installed. See the *Install datasets* section for more details.

### Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt
```

### Install datasets

```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:"path to the FixMatch"

# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create unlabeled datasets
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
wait

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 10 20 30 40 100 250 1000 4000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    for size in 400 1000 2500 10000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
    done
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
```

### ImageNet

Codebase for ImageNet experiments located in the [imagenet subdirectory](https://github.com/google-research/fixmatch/tree/master/imagenet).

## Running

### Setup

All commands must be ran from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

### Example

For example, training a FixMatch with 32 filters on cifar10 shuffled with `seed=3`, 40 labeled samples and 1
validation sample:
```bash
CUDA_VISIBLE_DEVICES=0 python fixmatch.py --filters=32 --dataset=cifar10.3@40-1 --train_dir ./experiments/fixmatch
```

Available labelled sizes are 10, 20, 30, 40, 100, 250, 1000, 4000.
For validation, available sizes are 1, 5000.
Possible shuffling seeds are 1, 2, 3, 4, 5 and 0 for no shuffling (0 is not used in practiced since data requires to be
shuffled for gradient descent to work properly).


#### Multi-GPU training
Just pass more GPUs and fixmatch automatically scales to them, here we assign GPUs 4-7 to the program:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --dataset=cifar10.3@40-1 --train_dir ./experiments/fixmatch
```

#### Flags

```bash
python fixmatch.py --help
# The following option might be too slow to be really practical.
# python fixmatch.py --helpfull
# So instead I use this hack to find the flags:
fgrep -R flags.DEFINE libml fixmatch.py
```

The `--augment` flag can use a little more explanation. It is composed of 3 values, for example `d.d.d`
(`d`=default augmentation, for example shift/mirror, `x`=identity, e.g. no augmentation, `ra`=rand-augment,
 `rac`=rand-augment + cutout):
- the first `d` refers to data augmentation to apply to the labeled example. 
- the second `d` refers to data augmentation to apply to the weakly augmented unlabeled example. 
- the third `d` refers to data augmentation to apply to the strongly augmented unlabeled example. For the strong
augmentation, `d` is followed by `CTAugment` for `fixmatch.py` and code inside `cta/` folder.



### Valid dataset names
```bash
for dataset in cifar10 svhn svhn_noextra; do
for seed in 0 1 2 3 4 5; do
for valid in 1 5000; do
for size in 10 20 30 40 100 250 1000 4000; do
    echo "${dataset}.${seed}@${size}-${valid}"
done; done; done; done

for seed in 1 2 3 4 5; do
for valid in 1 5000; do
    echo "cifar100.${seed}@10000-${valid}"
done; done

for seed in 1 2 3 4 5; do
for valid in 1 5000; do
    echo "stl10.${seed}@1000-${valid}"
done; done
echo "stl10.1@5000-1"
```


## Monitoring training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:

```bash
tensorboard.sh --port 6007 --logdir ./experiments
```

## Checkpoint accuracy

We compute the median accuracy of the last 20 checkpoints in the paper, this is done through this code:

```bash
# Following the previous example in which we trained cifar10.3@250-5000, extracting accuracy:
./scripts/extract_accuracy.py ./experiments/fixmatch/cifar10.d.d.d.3@40-1/CTAugment_depth2_th0.80_decay0.990/FixMatch_archresnet_batch64_confidence0.95_filters32_lr0.03_nclass10_repeat4_scales3_uratio7_wd0.0005_wu1.0/

# The command above will create a stats/accuracy.json file in the model folder.
# The format is JSON so you can either see its content as a text file or process it to your liking.
```

## Adding datasets
You can add custom datasets into the codebase by taking the following steps:

1. Add a function to acquire the dataset to `scripts/create_datasets.py` similar to the present ones, e.g. `_load_cifar10`. 
You need to call `_encode_png` to create encoded strings from the original images.
The created function should return a dictionary of the format 
`{'train' : {'images': <encoded 4D NHWC>, 'labels': <1D int array>},
'test' : {'images': <encoded 4D NHWC>, 'labels': <1D int array>}}` .
2. Add the dataset to the variable `CONFIGS` in `scripts/create_datasets.py` with the previous function as loader. 
You can now run the `create_datasets` script to obtain a tf record for it.
3. Use the `create_unlabeled` and `create_split` script to create unlabeled and differently split tf records as above in the *Install Datasets* section.
4. In `libml/data.py` add your dataset in the `create_datasets` function. The specified "label" for the dataset has to match
the created splits for your dataset. You will need to specify the corresponding variables if your dataset 
has a different # of classes than 10 and different resolution and # of channels than 32x32x3
5. In `libml/augment.py` add your dataset to the `DEFAULT_AUGMENT` variable. Primitives "s", "m", "ms" represent mirror, shift and mirror+shift. 

## Citing this work

```bibtex
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```
