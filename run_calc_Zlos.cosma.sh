#!/bin/bash --login
#SBATCH --ntasks 8
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --job-name=calc_Zlos_FLARES
#SBATCH --array=1-39
#SBATCH -t 0-2:30
###SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/std_Zlos_output.%J
#SBATCH -e logs/std_Zlos_error.%J

module purge
# module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.6.5
module load gnu_comp/10.2.0 openmpi/4.1.1 hdf5/1.10.6 pythonconda3/2020-02

output_folder="data"

## load your environment (must contain the eagle_IO module)
# source ./venv_fl/bin/activate
conda activate eagle3p9

### For FLARES galaxies, change ntasks as required
array=(011_z004p770 010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000 004_z011p000 003_z012p000 002_z013p000 001_z014p000 000_z015p000)

for ii in ${array[@]}
  do
    mpiexec -n 8 python3 calc_Zlos.py $SLURM_ARRAY_TASK_ID $ii FLARES $output_folder
done


### For PERIODIC boxes: REF and AGNdT9, change ntasks and time as required
# array=(002_z009p993 003_z008p988 004_z008p075 005_z007p050 006_z005p971 008_z005p037)
# array=(006_z005p971 008_z005p037 009_z004p485 z003p984)
# array=(012_z003p017)
#
# for ii in ${array[@]}
#   do
#     mpiexec -n 15 python3 calc_Zlos.py 00 $ii REF ./
# done

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
