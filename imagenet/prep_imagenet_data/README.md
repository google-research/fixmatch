# Preparing ImageNet dataset.

## How we generated data for semi-supervised training

To generate supervised data we randomply sampled 10% of ImageNet training data,
then we reshuffled this data and saved it into shards of TFRecords.
This procedure was repeated 5 times with different random seeds to produce 5 splits
of supervised data.

For unsupevised data we used entire ImageNet training set, reshuffled it and saved into shards of TFRecords.

## How to reproduce our data splits

1. You have to have original ImageNet data in TFRecord format.
   Follow https://cloud.google.com/tpu/docs/imagenet-setup to get it.

2. Make sure to have python 3, pip and virual env installed.

    ```
    sudo apt install python3-dev python3-virtualenv
    ```

3. Create python virtual environment and install all necessary dependencies:

   ```
   virtualenv -p python3 --system-site-packages ~/.venv3/imagenet_prep
   source ~/.venv3/imagenet_prep/bin/activate
   pip install tensorflow==2.1
   pip install absl-py
  
   # Install Apache Beam:
   pip install apache-beam
   # Google Cloud Dataflow users should use the following command:
   # pip install apache-beam[gcp]

   # Additional library for faster data processing
   sudo apt-get install libsnappy-dev
   pip install python-snappy
   ```

4. Set environment variable `${IMAGENET_DIR}` to the directory with original ImageNet data
   in TFRecord format:

   ```
   export IMAGENET_DIR=/path/to/original/imagenet
   ```

5. Create a separate directory where data for semi-supervised training will be saved.
   Set environment variable `${SSL_IMAGENET_DIR}` to the path to this directory:

   ```
   export SSL_IMAGENET_DIR=/desired/path/to/ssl/imagenet
   mkdir -p ${SSL_IMAGENET_DIR}
   ```

6. Run convenience script which generates ImageNet splits:

   ```
   # Running data processing pipeline using Apache Beam direct runner with 16 threads
   # See https://beam.apache.org/documentation/runners/direct/ for detaisl
   ./generate_ssl_imagenet.sh --runner=DirectRunner --direct_num_workers=16
   ```

   NOTE: processing of all data might take few hours. You can consider using 
   [Google Cloud Dataflow Runner](https://beam.apache.org/documentation/runners/dataflow/) to
   accelerate processing. Keep in mind that using Google Cloud Dataflow will need addtional setup and may incur additional charges.
