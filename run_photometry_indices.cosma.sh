#!/bin/bash --login
#SBATCH --ntasks 1
#SBATCH -A dp004
#SBATCH -p cosma7-shm
#SBATCH --job-name=derived_FLARES
#SBATCH --array=0-39
#SBATCH -t 0-01:00
#SBATCH -o logs/std_phot_indices_output.%J
#SBATCH -e logs/std_phot_indices_error.%J


module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.6.5

## load your environment (must contain the eagle_IO module)
source ./venv_fl/bin/activate

output_folder="data"

### For FLARES galaxies, change ntasks as required
array=(011_z004p770 010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000 004_z011p000 003_z012p000 002_z013p000 001_z014p000 000_z015p000)

for ii in ${array[@]}
  do
    python3 write_photometry_indices.py $SLURM_ARRAY_TASK_ID $ii FLARES $output_folder
done

# ### For PERIODIC boxes: REF and AGNdT9
# array=(002_z009p993 003_z008p988 004_z008p075 005_z007p050 006_z005p971 008_z005p037, 010_z003p984 011_z003p528, 012_z003p017)
#
# for ii in ${array[@]}
#   do
#     python3 write_photometry_indices.py 0 $ii REF $output_folder
# done

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
