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
#SBATCH --cpus-per-task=2

#SBATCH --mem=2G

## Allocate GPUs. Format is gpu:<type of GPU>:<# of GPUs>.
## The GPU type is optional. Available types are a40 and a100.
#SBATCH --gres=gpu:a100:1

## Maximum job run time. Format is days-HH:MM:SS.
#SBATCH --time=0-00:05:00

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

module load python-isal/0.11.0-GCCcore-10.3.0

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
EXPERIMENT_NAME="ecf_pyECT"
NUM_DIRECTIONS=16
NUM_TIMESTEPS=128
DEVICE="cuda"

OUTPUT_PATH="results/${EXPERIMENT_NAME}/${NUM_DIRECTIONS}_dirs_${NUM_TIMESTEPS}_timesteps_cuda.csv"

# Run the python script
python experiments/${EXPERIMENT_NAME}.py --image_path="data/test_images_1024.npz" --output_path=$OUTPUT_PATH --num_directions=$NUM_DIRECTIONS --num_timesteps=$NUM_TIMESTEPS --device=$DEVICE --batch_size=1
