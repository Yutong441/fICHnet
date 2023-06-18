#!/bin/bash

#SBATCH --job-name=CNN
#SBATCH --output=/PHShome/yc703/Documents/fICHnet/logs.txt
#SBATCH --nodes=1
#SBATCH --partition=Short
#SBATCH --qos=Short
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ychen146@mgh.harvard.edu

user='/PHShome/yc703'
project='fICHnet'

module use /apps/modulefiles/conversion
module load python/3.8
source $user/.bashrc
conda activate postICH_gpu
cd $user/Documents/$project
export PYTHONPATH=$PYTHONPATH:$user/Documents/$project

# get all the images into a single file
# img_dir=/data/rosanderson/ich_ct/processed2/original
# save_dir=$(pwd)/results
# for i in $(ls $img_dir)
# do
#     echo $img_dir/$i >> $save_dir/"img_path.txt" 
# done

# running model
python predict.py --img_paths=results/img_path.txt \
    --model_path=models/ \
    --save_dir=results/ \
    --device="cuda:0"
