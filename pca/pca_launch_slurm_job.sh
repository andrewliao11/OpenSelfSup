#!/bin/bash

N_CLASSES=$1
FILENAME=$2
SPLIT=$3
JOB_NAME=$4
d=`date +%Y-%m-%d`
j_dir=./pca_outputs/$d/$JOB_NAME
cmd="python pca_dim_reduction.py --n_classes $N_CLASSES --filename $FILENAME --split $SPLIT"


# build slurm script
mkdir -p $j_dir/scripts
mkdir -p $j_dir/log
mkdir -p $j_dir/outputs
echo "#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${j_dir}/log/%j.out
#SBATCH --error=${j_dir}/log/%j.err
#SBATCH --qos=nopreemption
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
bash ${j_dir}/scripts/${JOB_NAME}.sh
" > $j_dir/scripts/${JOB_NAME}.slrm

# build bash script
echo -n "#!/bin/bash
${cmd} --work_dir=${j_dir}/outputs
" > $j_dir/scripts/${JOB_NAME}.sh

sbatch $j_dir/scripts/${JOB_NAME}.slrm
