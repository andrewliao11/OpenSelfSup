root="/h/andrewliao/UofT/cvpr2021/OpenSelfSup/outputs/2020-08-28/"
task="attribute"


for feat in 'train-moco_r50_v2-58f10cfe.pth/outputs/20200828_005920/features/train/moco_r50_v2-58f10cfe.pth_feat1.npy' 'train-rotation_r50-cfab8ebb.pth/outputs/20200828_005920/features/train/rotation_r50-cfab8ebb.pth_feat1.npy' 'train-byol_r50-e3b0c442.pth/outputs/20200828_005910/features/train/byol_r50-e3b0c442.pth_feat1.npy' 'train-relative_loc_r50-342c9097.pth/outputs/20200828_005910/features/train/relative_loc_r50-342c9097.pth_feat1.npy' 'train-simclr_r50_bs256_ep200-4577e9a6.pth/outputs/20200828_005920/features/train/simclr_r50_bs256_ep200-4577e9a6.pth_feat1.npy'; do
    fit_fn="svm"
    job_name=$(basename $feat)-$task-$fit_fn
    bash launch_slurm_job.sh $root$feat $fit_fn $job_name $task cpu 0
done


for feat in 'train-moco_r50_v2-58f10cfe.pth/outputs/20200828_005920/features/train/moco_r50_v2-58f10cfe.pth_feat1.npy' 'train-rotation_r50-cfab8ebb.pth/outputs/20200828_005920/features/train/rotation_r50-cfab8ebb.pth_feat1.npy' 'train-byol_r50-e3b0c442.pth/outputs/20200828_005910/features/train/byol_r50-e3b0c442.pth_feat1.npy' 'train-relative_loc_r50-342c9097.pth/outputs/20200828_005910/features/train/relative_loc_r50-342c9097.pth_feat1.npy' 'train-simclr_r50_bs256_ep200-4577e9a6.pth/outputs/20200828_005920/features/train/simclr_r50_bs256_ep200-4577e9a6.pth_feat1.npy'; do
    fit_fn="linear"
    job_name=$(basename $feat)-$task-$fit_fn
    bash launch_slurm_job.sh $root$feat $fit_fn $job_name $task p100 1
done