# TrypanoDeepScreen API

This repository provides a Docker-based API to train and predict compound bioactivity using deep learning, based on [ahmetrifaioglu's DEEPScreen](https://github.com/cansyl/DEEPScreen) published in [Chemical Science (2020)](https://doi.org/10.1039/C9SC03414E).

TrypanoDEEPScreen is implemented using PyTorch Lightning and leverages Ray Tune for efficient hyperparameter search. GPU usage and parallelization (when available) are completely handled by the pipeline. It also fixes some bugs from the original predictor, and implements enseambleing.

## What is Docker?

Docker is a platform that allows applications to run inside isolated environments called containers. A container includes everything needed to execute an application, such as libraries, dependencies, and system configurations, ensuring consistency across different computing environments.

TrypanoDeepScreen runs inside a Docker container, meaning that all its dependencies are pre-installed and configured within a self-contained environment. This ensures reproducibility and eliminates compatibility issues.


Docker containers are ephemeral, meaning that any data stored inside them will be lost once the container stops or is removed. To persist data, bind mounts are used to link directories from the host machine to the container. This ensures that:

Input files can be accessed by the container.

Trained models and generated predictions are stored outside the container and persist after execution.

## Requirements
- Docker. [Install](https://docs.docker.com/get-started/get-docker/)
- Properly formatted input data

## Workflow Explanation

The TrypanoDeepScreen API follows a structured workflow comprising two main stages:

### 1. Docker Setup

You have two ways to set up the Docker environment for TrypanoDeepScreen:

- **Option A: Pull pre-built Docker image from DockerHub (RECOMENDED)**
  ```bash
  docker pull sebastianjinich/trypanodeepscreen:latest
  ```
  *(Replace `<dockerhub_username>` with the actual username hosting the image.)*

- **Option B: Build the Docker image from scratch**
  ```bash
  docker build -t trypanodeepscreen .
  ```

### 2. Running the Container

Execute TrypanoDeepScreen by running the Docker container with mounted directories. The general command is:

```bash
docker run -it \
    --mount type=bind,src=/path/to/data,dst=/root/trypanodeepscreen/data \
    --mount type=bind,src=/path/to/experiments,dst=/root/trypanodeepscreen/trained_models \
    --mount type=bind,src=/path/to/predictions,dst=/root/trypanodeepscreen/predictions \
    --mount type=bind,src=/path/to/config,dst=/root/trypanodeepscreen/config \
    --shm-size=15gb \
    sebastianjinich/trypanodeepscreen:latest <mode> [options]
```
### Operational Workflow

- **Training**: Run training mode with labeled datasets to train a new model.
- **Prediction**: Run prediction mode using previously trained models to generate bioactivity predictions for new compounds.


## Docker Setup

Build the Docker image:

```bash
cd trypanodeepscreen_container # Where Dockerfile is located
docker build -t trypanodeepscreen .
```
This will build the container from scratch, install all dependencies automaticaly, and get it ready to be run.

```bash
docker images # run this to find out if the docker was succesfully created
```


## Running Trypanodeepscreen in the container

Use bind mounts to connect local directories to the Docker container. Example:

```bash
docker run -it \
    --mount type=bind,src=/path/to/data,dst=/root/trypanodeepscreen/data \
    --mount type=bind,src=/path/to/experiments,dst=/root/trypanodeepscreen/trained_models \
    --mount type=bind,src=/path/to/predictions,dst=/root/trypanodeepscreen/predictions \
    --mount type=bind,src=./config,dst=/root/trypanodeepscreen/config\
    --shm-size=15gb \
    sebastianjinich/trypanodeepscreen:latest <mode> [options]
```
Replace paths accordingly.

### Directory Mapping

When running the container, the following directories are mounted from the host system to the Docker environment. This means that any files placed in these directories on the host machine will be accessible inside the container, and any outputs generated inside the container will be saved in these locations on the host system:

- `/data`: This directory is used to provide input files required for training or prediction. The source directory specified in the `--mount` option must contain the necessary input files.
- `/experiments`: This is where trained models are stored. If running a training session, the resulting model files will be saved here. If running predictions, the models stored in this directory will be used.
- `/predictions`: This directory is used to store the output files generated during prediction. Any prediction results will be written to this directory inside the container, and they will also be accessible in the corresponding mounted directory on the host system.
- `/config`: This directory is used to store the config.yml file. Only mount this directory if you want to modify default configurations.

By mounting directories in this way, the container can read input data and write output results in a structured manner without requiring direct modifications inside the container itself.

Replace paths accordingly when mounting directories.

### Modes of running docker

Docker containers can be executed in different modes depending on the use case:

- **Interactive Mode (**`-it`**)**: Runs the container in an interactive session where commands can be executed manually. This is useful for looking into the process output and development.
  ```bash
  docker run -it trypanodeepscreen ...
  ```
- **Detached Mode (**`-d`**)**: Runs the container in the background. This is useful for long-running processes where continuous interaction is not needed.
  ```bash
  docker run -d trypanodeepscreen ...
  ```

To check running containers in detached mode:

```bash
docker ps
```

To stop a running container:

```bash
docker stop <container_id>
```

### Example Scripts

There are example scripts available in the repository to run a training session and a prediction:

- `train_run_example.sh`: Example script to execute a training session.
- `prediction_run_example.sh`: Example script to execute a prediction.

These scripts include the necessary commands to facilitate execution. Read full README for a better understanding of the usage.

```
bash train_run_example.sh
```
## GPU Usage (Optional)

If you have an NVIDIA GPU and wish to use it for training or prediction to accelerate computation, follow the steps below.

### 1. Install NVIDIA Container Toolkit

To allow Docker containers to access your GPU hardware, you must install the NVIDIA Container Toolkit. Follow the official installation guide:

**NVIDIA Container Toolkit Installation Guide:**
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Make sure to complete all the steps for your operating system. 

### 2. Using GPU with TrypanoDeepScreen

Once the toolkit is installed and GPU is available to Docker, you can run TrypanoDeepScreen with GPU support by adding the `--gpus "all"` flag to the Docker command:

```bash
docker run -it \
    --mount type=bind,src=./data,dst=/root/trypanodeepscreen/data \
    --mount type=bind,src=./trained_models,dst=/root/trypanodeepscreen/trained_models \
    --mount type=bind,src=./config,dst=/root/trypanodeepscreen/config \
    --mount type=bind,src=./predictions,dst=/root/trypanodeepscreen/predictions \
    --shm-size=15gb \
    --gpus "all" \
    sebastianjinich/trypanodeepscreen:latest \
       train \
            --data_train_val_test data/train_data_example.csv \
            --target_name trypano_experiment_example \
            --experiment_result_path trained_models/
```

> **Note:** Use the `--gpus "all"` flag **only if your system has compatible NVIDIA GPUs and the NVIDIA Container Toolkit is properly installed.** Otherwise, omit this flag.

---

## Detailed API running

### 1. Train Mode
Trains the model with hyperparameter tuning.

**Usage:**

```bash
train \
    --data_train_val_test path/to/data.csv \
    [--target_name experiment_name] \
    [--experiment_result_path /path/to/models] \
```


**Required CSV format:**
- `comp_id`: Unique compound identifier.
- `smiles`: Compound SMILES structure.
- `bioactivity`: Binary bioactivity label (0 or 1).
- `data_split`: Data splits (`train`, `validation`, `test`).

```
comp_id        bioactivity   data_split   smiles
CHEMBL101747   1            test         CCN1CCC(CC(=O)Nc2n...
CHEMBL101804   1            train        O=C(Nc1n[nH]c2nc(-...
CHEMBL102714   1            train        Cn1cc(C2=C(c3ccc(C...
CHEMBL1078178  0            test         N#CCNC(=O)c1ccc(-c...
CHEMBL1079175  0            test         NC1(c2ccc(-c3nc4cc...
```

### Example:

```
docker run -d\
    --mount type=bind,src=./ml,dst=/root/trypanodeepscreen/ml\
    --mount type=bind,src=./data,dst=/root/trypanodeepscreen/data\
    --mount type=bind,src=./trained_models,dst=/root/trypanodeepscreen/trained_models\
    --mount type=bind,src=./config,dst=/root/trypanodeepscreen/config\
    --mount type=bind,src=./predictions,dst=/root/trypanodeepscreen/predictions\
    --shm-size=15gb\
    sebastianjinich/trypanodeepscreen:latest \
        train \
            --data_train_val_test data/train_data_example.csv \
            --target_name trypano_experiment_example \
            --experiment_result_path trained_models/ 
```

### Prediction Mode

Generates predictions using trained ensemble models.

**Usage:**

```bash
predict \
    --model_folder_path /path/to/trained_models \
    --data_input_prediction /path/to/input.csv \
    [--metric_ensambled_prediction val_auroc] \
    [--result_path_prediction_csv /path/to/output.csv] \
    [--n_checkpoints 26] \
    [--config_file /path/to/config.yml]
```

**Required CSV format:**
- `comp_id`: Compound unique identifier.
- `smiles`: SMILES string.

### Example:

```bash
docker run -it\
    --mount type=bind,src=./data,dst=/root/trypanodeepscreen/data\
    --mount type=bind,src=./trained_models,dst=/root/trypanodeepscreen/trained_models\
    --mount type=bind,src=./config,dst=/root/trypanodeepscreen/config\
    --mount type=bind,src=./predictions,dst=/root/trypanodeepscreen/predictions\
    --shm-size=15gb\
    sebastianjinich/trypanodeepscreen:latest \
        predict \
            --model_folder_path trained_models/trypano_experiment_example \
            --data_input_prediction data/predict_data_example.csv \
            --result_path_prediction_csv predictions/prediction_results_example.csv \
            --metric_ensambled_prediction val_auroc \
            --n_checkpoints 26

```

### Prediction Output Format

The prediction results are saved as a CSV file in the directory mapped to `/root/trypanodeepscreen/predictions`. The output file contains the predictions from all models in the ensemble. Each row corresponds to a compound, with columns representing the predictions made by individual models. The final column, ``, contains the ensemble model's aggregated score, which is the final prediction.

#### Example Output:

```
comp_id        model_1   model_2   ...  prediction_mean_score
cefmenoxime    0.407573  0.724858  ...  0.566215
ulifloxacin    0.060150  0.573115  ...  0.316633
cefotiam       0.967878  0.793163  ...  0.880521
ceftriaxone    0.640682  0.758880  ...  0.699781
balofloxacin   0.107408  0.573296  ...  0.340352
```

## Optional Configuration File

Use a YAML file to override default parameters. Modify the config/config.yml file in the config folder, and mount the folder to the docker.

```yaml
mol_draw_options: # Settings for molecule structure visualization
  atomLabelFontSize: 55  # Font size for atom labels
  dotsPerAngstrom: 100   # Resolution for rendering molecules
  bondLineWidth: 1       # Width of molecular bonds in the rendering

img_size: 200  # Size of molecule images used in training

use_tmp_imgs: False  # Whether to store generated images persistently or use temporary storage

hyperparameters_search: # Hyperparameters for model training
  fully_layer_1: [16, 32, 128, 256, 512]  # First hidden layer sizes
  fully_layer_2: [16, 32, 128, 256, 512]  # Second hidden layer sizes
  learning_rate: [0.0005, 0.0001, 0.005, 0.001, 0.01]  # Learning rates to test
  batch_size: [32, 64]  # Mini-batch sizes for training
  drop_rate: [0.3, 0.5, 0.6, 0.8]  # Dropout rates for regularization

hyperparameters_search_setup: # Configuration for hyperparameter optimization
  max_epochs: 100  # Maximum training epochs
  grace_period: 13  # Number of epochs before pruning underperforming trials
  metric_to_optimize: "val_auroc"  # Performance metric to optimize
  optimize_mode: "max"  # Whether to maximize or minimize the metric
  num_samples: 350  # Number of hyperparameter combinations to test
  asha_reduction_factor: 4  # Reduction factor for adaptive stopping
  number_ckpts_keep: 2  # Number of model checkpoints to retain

max_cpus: 20  # Maximum number of CPU cores to use
max_gpus: 2   # Maximum number of GPUs to use
```

---
## Notes
Contact the repository [maintainer](mailto:sebij1910@gmail.com) for further assistance. 

