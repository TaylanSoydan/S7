# Codebase README

## Prerequisites

1. **Data Folder Setup**:
   - Ensure the following data folder structure is set up:
     ```
     dvs128/
     eigenworms/
     eventdatasets/
     long-range-arena/
     person/
     physionet/
     ptb/
     walker/
     wikitext2/
     wikitext103/
     ```

2. **Environment Setup**:
   - Create and activate the Conda environment:
     ```bash
     conda create --name testenv python=3.9.19
     conda activate testenv
     ```
   - Install the required Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Configuration

1. **System Configuration**:
   - Update the `configs/system/local.yaml` file to point to your local `eventdatasets` folder:
     ```yaml
     data_dir: /data/storage/tsoydan/data/eventdatasets # Point it to your own local eventdatasets folder
     ```

2. **Task and Model Configuration**:
   - Modify the YAML configuration files located inside the `configs/` folder to specify the task and model. For example, update settings for `wikitext2` or other tasks as needed.

## Running Experiments

1. **Run Training**:
   - Execute the training script with the specified task. Example:
     ```bash
     CUDA_VISIBLE_DEVICES=0 python3 run_training.py task=text
     ```

2. **Run a Weights & Biases Sweep**:
   - Execute a sweep using the following command:
     ```bash
     wandb sweep image_sweep.yaml
     CUDA_VISIBLE_DEVICES=0 wandb agent taylansoydan/event-ssm/la75bvy4
     ```

3. **Evaluate a Checkpoint**:
   - Use the following command to evaluate a checkpoint:
     ```bash
     CUDA_VISIBLE_DEVICES=0 python run_evaluation.py task=text checkpoint=/data/old_home/tsoydan/RPG/event-ssm/checkpoints/best_text/checkpoints
     ```
    Make sure that the model and task config parameters are in accordance with the checkpoint model. For example if the checkpoint d_ssm = 70, /configs/model/listops/small.yaml d_ssm should also equal 70.

