#!/bin/bash
#BSUB -J CNN
#BSUB -q normal
#BSUB -n 4
#BSUB -R rusage[mem=10000]

user='/PHShome/yc703'
project='fICHnet'

module load python/3.8
source $user/.bashrc
conda activate postICH
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
    --device="cpu"
