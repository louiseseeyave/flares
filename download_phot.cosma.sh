#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --job-name=phot_write
#SBATCH -t 0-08:00
#SBATCH --array=0-39%10
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
###SBATCH --ntasks-per-node=1
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J


module purge
module load python/3.6.5 gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3

source ./venv_fl/bin/activate

## Change the argument to the script to Luminosity or Flux; FLARES or REF or AGNdT9
## as required

### For FLARES galaxies, change ntasks as required
array=(011_z004p770 010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000 004_z011p000 003_z012p000 002_z013p000 001_z014p000 000_z015p000)


for ii in ${array[@]}
  do
    python3 download_photproperties.py $SLURM_ARRAY_TASK_ID $ii FLARES Luminosity data
    python3 download_photproperties.py $SLURM_ARRAY_TASK_ID $ii FLARES SED data
    python3 download_photproperties.py $SLURM_ARRAY_TASK_ID $ii FLARES Lines data
    python3 download_photproperties.py $SLURM_ARRAY_TASK_ID $ii FLARES Flux data
done


### For PERIODIC boxes: REF and AGNdT9

# array=(002_z009p993 003_z008p988 004_z008p075 005_z007p050 006_z005p971 008_z005p037 010_z003p984 012_z003p017)
# array=(008_z005p037 010_z003p984 012_z003p017)
#
# for ii in ${array[@]}
#   do
#     python3 download_photproperties.py 00 $ii REF Luminosity ./
#     # python3 download_photproperties.py 00 $ii REF SED ./
#     python3 download_photproperties.py 00 $ii REF Lines ./
#     python3 download_photproperties.py 00 $ii REF Flux ./
# done

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
