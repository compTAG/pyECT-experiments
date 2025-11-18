#!/bin/bash
##
## GPU priority partition sbatch example.
##
## Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
## All other lines are read by the shell.

## Specify the account to use; you must use your priority account.
#SBATCH --account=priority-brittanyfasy

## Use the gpupriority partition.
#SBATCH --partition=gpupriority

## Number of CPUs to allocate. Note that in the slurm, CPUs correspond to
## logical processors, not physical cores.
#SBATCH --cpus-per-task=1

#SBATCH --mem=2G

## Allocate GPUs. Format is gpu:<type of GPU>:<# of GPUs>.
## The GPU type is optional. Available types are a40 and a100.
#SBATCH --gres=gpu:a100:1

## Maximum job run time. Format is days-HH:MM:SS.
#SBATCH --time=0-00:59:00

## Set job name.
#SBATCH --job-name=gpupriority-example

## Set output log and error log filenames. 
## %j is the job number and %x is the job name set above 
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

## Remove one # and fill out the following lines if you want to receive emails.
##SBATCH --mail-user=eli.quist@student.montana.edu      # enter your email to recieve email notifications
##SBATCH --mail-type=ALL                 # specify conditions on which to recieve emails

## Run 'man sbatch' for more information on the options above.
## Below, enter commands needed to execute your workload

# Print out the date.
date
# Print which partition we are using.
echo "partition = ${SLURM_JOB_PARTITION}"
# Print the compute node's name.
echo "Hello from $(hostname -s)"
# Print which gpu we are using.
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}"
# Print the date again.
date

module load Python/3.10.8-GCCcore-12.2.0

# Clone the latest version of the pyECT-experiments repo
if [ ! -d "$HOME/pyECT-experiments" ]; then
    git clone git@github.com:compTAG/pyECT-experiments.git "$HOME/pyECT-experiments"
else
    cd "$HOME/pyECT-experiments" || exit 1
    git pull origin main

fi

# move to pyECT experiments repo
cd $HOME/pyECT-experiments || exit 1

## create new virtualenv if doesn't exist, otherwise load 'pyect-experiments' env
if [ ! -d "$HOME/pyect-experiments-env" ]; then
    python3 -m venv "$HOME/pyect-experiments-env"
fi

source "$HOME/pyect-experiments-env/bin/activate"
which python
python --version
which pip
pip install --upgrade pip
cat requirements.txt
pip install -r requirements.txt
pip list

# check if datatest_images_1024.npz file exists, if not create it and generate test images
if [ ! -f "data/test_images_1024.npz" ]; then
    cd data
    python generate_test_images.py
    cd ..
fi

## ----- EXPERIMENTS -----

# Create variables for number of directions and timesteps
INVARIANT="wect"
IMPLEMENTATION_NAME="pyECT"
DEVICE="cuda"
DATASET="test_images_1024"

DIRECTIONS=(1 2 4 8 16 32 64 128)
TIMESTEPS=(10 100 1000 10000)

for NUM_DIRECTIONS in "${DIRECTIONS[@]}"; do
  for NUM_TIMESTEPS in "${TIMESTEPS[@]}"; do
    
    OUTPUT_PATH="results/${DATASET}/${INVARIANT}/${IMPLEMENTATION_NAME}/${NUM_DIRECTIONS}_dirs_${NUM_TIMESTEPS}_timesteps_${DEVICE}.csv"

    # ensure directory exists
    DIR_NAME=$(dirname "$OUTPUT_PATH")
    mkdir -p "$DIR_NAME"

    echo "Running: directions=$NUM_DIRECTIONS, timesteps=$NUM_TIMESTEPS"

    python experiments/main.py \
      --data_path="data/${DATASET}.npz" \
      --data_type=image \
      --invariant="${INVARIANT}" \
      --implementation_name="${IMPLEMENTATION_NAME}" \
      --output_path="${OUTPUT_PATH}" \
      --num_directions="${NUM_DIRECTIONS}" \
      --num_timesteps="${NUM_TIMESTEPS}" \
      --device="${DEVICE}" \
      --batch_size=1 \
      --cores=1

  done
done
