#!/bin/bash
#SBATCH --ntasks 8
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --job-name=get_FLARES
####SBATCH --array=0-30%10
###SBATCH --array=0-3%1
#SBATCH -t 0-00:04
###SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J

module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.6.5

#arg1 is the region number (in case of FLARES, leave it as some number for periodic boxes), arg2 is the relevant tag (snap number), arg3 is FLARES/REF/AGNdT9 and arg4 is the text file with the relevant arrys to be written into the hdf5 file
#In case of Periodic boxes, set the ntasks number to a cube to make a proper division of the box


source ./venv_fl/bin/activate

### For FLARES galaxies, change ntasks as required
# array=(011_z004p770 010_z005p000 009_z006p000) #008_z007p000 007_z008p000 006_z009p000 005_z010p000 004_z011p000 003_z012p000 002_z013p000 001_z014p000 000_z015p000)

# mpiexec -n 16 -m mpi4py python3 download_methods.py $SLURM_ARRAY_TASK_ID ${array[0]} FLARES req_arrs.txt
# mpiexec -n 16 -m mpi4py python3 download_methods.py $SLURM_ARRAY_TASK_ID ${array[1]} FLARES req_arrs.txt
# mpiexec -n 16 -m mpi4py python3 download_methods.py $SLURM_ARRAY_TASK_ID ${array[2]} FLARES req_arrs.txt
# mpiexec -n 16 -m mpi4py python3 download_methods.py $SLURM_ARRAY_TASK_ID ${array[3]} FLARES req_arrs.txt
# mpiexec -n 16 -m mpi4py python3 download_methods.py $SLURM_ARRAY_TASK_ID ${array[4]} FLARES req_arrs.txt
# mpiexec -n 16 -m mpi4py python3 download_methods.py $SLURM_ARRAY_TASK_ID ${array[5]} FLARES req_arrs.txt


### For PERIODIC boxes: REF and AGNdT9, change ntasks and time as required (REF at z=5 required ~1.hr)
# array=(002_z009p993 003_z008p988 004_z008p075 005_z007p050 006_z005p971 008_z005p037)
# array=(006_z005p971 008_z005p037 009_z004p485 z003p984)

# mpiexec -n 64 python3 -m mpi4py download_methods.py $SLURM_ARRAY_TASK_ID ${array[$SLURM_ARRAY_TASK_ID]} REF req_arrs.txt

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
