splits=("train" "val")
config="configs/benchmarks/linear_classification/imagenet/r50_last.py"
weight=("byol_r50-e3b0c442.pth" "relative_loc_r50-342c9097.pth" "simclr_r50_bs256_ep200-4577e9a6.pth" "moco_r50_v2-58f10cfe.pth" "rotation_r50-cfab8ebb.pth")
root="/scratch/hdd001/home/andrewliao/openselfsup/weights/"


for s in "train" "val"; do
    for w in "byol_r50-e3b0c442.pth" "relative_loc_r50-342c9097.pth" "simclr_r50_bs256_ep200-4577e9a6.pth" "moco_r50_v2-58f10cfe.pth" "rotation_r50-cfab8ebb.pth"; do
        job_name=$s-$w
        bash launch_slurm_job.sh $s $config $root$w $job_name
    done
done
