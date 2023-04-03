#!/bin/bash
#SBATCH --job-name=fairface-test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=eh0560@princeton.edu # change this email address.

module purge
module load anaconda3/2022.10
conda activate torch-env

python predict.py --input_csv "/scratch/gpfs/eh0560/FairFace/test/test_imgs.csv" --detected_faces_output "/scratch/gpfs/eh0560/FairFace/test/detected_faces" --prediction_output "/scratch/gpfs/eh0560/FairFace/test/test_outputs.csv"
