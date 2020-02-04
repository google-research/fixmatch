#!/bin/bash
#
# Script which generates splits with semi-supervised Imagenet data.
#

if [ -z ${IMAGENET_DIR} ]; then
  echo "Variable IMAGENET_DIR has to be set."
  exit 1
fi

if [ -z ${SSL_IMAGENET_DIR} ]; then
  echo "Variable SSL_IMAGENET_DIR has to be set."
  exit 1
fi

# Generating labelled splits with 10% of the data
for seed in 1 2 3 4 5; do
  python -B save_sharded_data.py \
    --input_dir="${IMAGENET_DIR}" \
    --sharding_file="data_splits/files2shards_train_size128116_split${seed}.txt" \
    --output_file_prefix="${SSL_IMAGENET_DIR}/train128116.${seed}" \
    -- "$@"
    exit 1
done

# Generating re-shuffled split with all Imagenet data.
# It will be used as unlabelled data.
  python -B save_sharded_data.py \
    --input_dir="${IMAGENET_DIR}" \
    --sharding_file="data_splits/files2shards_train}.txt" \
    --output_file_prefix="${SSL_IMAGENET_DIR}/train" \
    -- "$@"

# Copy validation data to output directory
cp ${IMAGENET_DIR}/validation-* ${SSL_IMAGENET_DIR}
