#!/bin/bash
filename=$1
fit_fn=$2
job_name=$3
task=$4
partition=$5
n_gpus=$6

d=`date +%Y-%m-%d`
j_dir=./outputs/$d/$job_name
cmd="python main.py --filename $filename --task $task --fit_fn $fit_fn"


# build slurm script
mkdir -p $j_dir/scripts
mkdir -p $j_dir/log
mkdir -p $j_dir/outputs
echo "#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${j_dir}/log/%j.out
#SBATCH --error=${j_dir}/log/%j.err
#SBATCH --qos=nopreemption
#SBATCH --partition=$partition
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:$n_gpus
#SBATCH --nodes=1
bash ${j_dir}/scripts/${job_name}.sh
" > $j_dir/scripts/${job_name}.slrm

# build bash script
echo -n "#!/bin/bash
${cmd} --work_dir=${j_dir}/outputs
" > $j_dir/scripts/${job_name}.sh

sbatch $j_dir/scripts/${job_name}.slrm
