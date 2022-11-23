#!/bin/bash --login
#SBATCH --ntasks 1
#SBATCH -A dp004
#SBATCH -p cosma7-shm
#SBATCH --job-name=derived_FLARES
#SBATCH --array=0-39
#SBATCH -t 0-02:00
#SBATCH -o logs/std_derived_output.%J
#SBATCH -e logs/std_derived_error.%J


module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.6.5
# module load gnu_comp/10.2.0 openmpi/4.1.1 hdf5/1.10.6 pythonconda3/2020-02

# arg1 is the region number (in case of FLARES, leave it as some number for periodic boxes), arg2 is the relevant tag (snap number), arg3 is FLARES/REF/AGNdT9, arg4 is the text file with the relevant arrys to be written into the hdf5 file and agr5 is the folder you want the files to be saved

output_folder="data"

## load your environment (must contain the eagle_IO module)
# source ./venv_fl/bin/activate
# conda activate eagle3p9
source /cosma7/data/dp004/dc-seey1/venvs/pyenv3.9/bin/activate

### For FLARES galaxies, change ntasks as required
array=(011_z004p770 010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000 004_z011p000 003_z012p000 002_z013p000 001_z014p000 000_z015p000)

for ii in ${array[@]}
  do
    python3 write_ionising_properties.py $SLURM_ARRAY_TASK_ID $ii FLARES $output_folder
done

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
