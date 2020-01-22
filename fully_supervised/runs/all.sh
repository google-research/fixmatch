#!/usr/bin/env bash

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fully supervised baseline without mixup (not shown in paper since Mixup is better)
python fully_supervised/fs_baseline.py --train_dir experiments/fs --dataset=cifar10-1 --wd=0.02 --smoothing=0.001
python fully_supervised/fs_baseline.py --train_dir experiments/fs --dataset=cifar100-1 --wd=0.02 --smoothing=0.001
python fully_supervised/fs_baseline.py --train_dir experiments/fs --dataset=svhn-1 --wd=0.002 --smoothing=0.01
python fully_supervised/fs_baseline.py --train_dir experiments/fs --dataset=svhn_noextra-1 --wd=0.002 --smoothing=0.01

# Fully supervised Mixup baselines (in paper)
# Uses default parameters: --wd=0.002 --beta=0.5
python fully_supervised/fs_mixup.py --train_dir experiments/fs --dataset=cifar10-1
python fully_supervised/fs_mixup.py --train_dir experiments/fs --dataset=svhn-1
python fully_supervised/fs_mixup.py --train_dir experiments/fs --dataset=svhn_noextra-1

# Fully supervised Mixup baselines on 26M parameter large network (in paper)
# Uses default parameters: --wd=0.002 --beta=0.5
python fully_supervised/fs_mixup.py --train_dir experiments/fs --dataset=cifar10-1 --filters=135
python fully_supervised/fs_mixup.py --train_dir experiments/fs --dataset=cifar100-1 --filters=135
