#!/bin/bash
##
## Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
## All other lines are read by the shell.
##
## Basic parameters
##
#SBATCH --account=group-   	# specify the account to use if using a priority partition
#SBATCH --partition=test                # queue partition to run the job in
#SBATCH --cpus-per-task=1               # number of cores to allocate
#SBATCH --mem=2G                        # ammount of Memory allocated
#SBATCH --time=0-00:59:00               # maximum job run time in days-hours:minutes:secconds
#SBATCH --job-name=example              # job name
#SBATCH --output=example-%j.out         # standard output from job
#SBATCH --error=example-%j.err          # standard error from job
#SBATCH --mail-user=email@mail.com      # enter your email to recieve email notifications
#SBATCH --mail-type=ALL                 # specify conditions on which to recieve emails

date                                    # print out the date
echo "hello from $(hostname -s)"        # print a message from the compute node
date                                    # print the date again

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
INVARIANT="ecf"
IMPLEMENTATION_NAME="pyECT"
DEVICE="cpu"
DATASET="test_images_1024"
CORES=1

DIRECTIONS=(0)
TIMESTEPS=(10 100 1000 10000)

for NUM_DIRECTIONS in "${DIRECTIONS[@]}"; do
  for NUM_TIMESTEPS in "${TIMESTEPS[@]}"; do

    OUTPUT_PATH="results/${DATASET}/${INVARIANT}/${IMPLEMENTATION_NAME}/${NUM_DIRECTIONS}_dirs_${NUM_TIMESTEPS}_timesteps_${DEVICE}_${CORES}_cores.csv"

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
      --cores=${CORES}

  done
done