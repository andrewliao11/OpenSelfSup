root="/h/andrewliao/UofT/cvpr2021/OpenSelfSup/outputs/2020-08-28/"


for c in 100 500 1000; do
    for f in 'train-moco_r50_v2-58f10cfe.pth/outputs/20200828_005920/features/train/moco_r50_v2-58f10cfe.pth_feat1.npy' 'train-rotation_r50-cfab8ebb.pth/outputs/20200828_005920/features/train/rotation_r50-cfab8ebb.pth_feat1.npy' 'train-byol_r50-e3b0c442.pth/outputs/20200828_005910/features/train/byol_r50-e3b0c442.pth_feat1.npy' 'train-relative_loc_r50-342c9097.pth/outputs/20200828_005910/features/train/relative_loc_r50-342c9097.pth_feat1.npy' 'train-simclr_r50_bs256_ep200-4577e9a6.pth/outputs/20200828_005920/features/train/simclr_r50_bs256_ep200-4577e9a6.pth_feat1.npy'; do
        split="train"
        job_name=$split-$c-$(basename "${f}")
        bash pca_launch_slurm_job.sh $c $root$f $split $job_name
    done
done


for c in 100 500 1000; do
    for f in 'val-byol_r50-e3b0c442.pth/outputs/20200828_011314/features/val/byol_r50-e3b0c442.pth_feat1.npy' 'val-simclr_r50_bs256_ep200-4577e9a6.pth/outputs/20200828_012143/features/val/simclr_r50_bs256_ep200-4577e9a6.pth_feat1.npy' 'val-relative_loc_r50-342c9097.pth/outputs/20200828_011732/features/val/relative_loc_r50-342c9097.pth_feat1.npy' 'val-rotation_r50-cfab8ebb.pth/outputs/20200828_012756/features/val/rotation_r50-cfab8ebb.pth_feat1.npy' 'val-moco_r50_v2-58f10cfe.pth/outputs/20200828_012555/features/val/moco_r50_v2-58f10cfe.pth_feat1.npy'; do
        split="val"
        job_name=$split-$c-$(basename "${f}")
        bash pca_launch_slurm_job.sh $c $root$f $split $job_name
    done
done
