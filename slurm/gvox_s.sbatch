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

#SBATCH --mem=64G

## Allocate GPUs. Format is gpu:<type of GPU>:<# of GPUs>.
## The GPU type is optional. Available types are a40 and a100.
#SBATCH --gres=gpu:a100:1

## Maximum job run time. Format is days-HH:MM:SS.
#SBATCH --time=0-00:59:00

## Set job name.
#SBATCH --job-name=gvox-s

## Set output log and error log filenames. 
## %j is the job number and %x is the job name set above 
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

## Remove one # and fill out the following lines if you want to receive emails.
##SBATCH --mail-user=     # enter your email to recieve email notifications
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

## ----- EXPERIMENT -----

OUTPUT_PATH="results/gvox_s.csv"
mkdir -p results

echo "Running gvox_s.py throughput benchmark on CUDA"

python experiments/gvox_s.py \
  --device cuda \
  --output "${OUTPUT_PATH}"

echo "Done. Results written to ${OUTPUT_PATH}"
