#!/bin/sh

# # Run the main.py script
# screen -S "screen" -d -m taskset --cpu-list 0 python main.py

# # Make copies of assets and rename the copy as assets_last_github_commit_ID
# last_github_commit_ID="eed3553"
# models="base_model base_model_FE base_model_FE-CONTRAST-ONLY mask_model mask_contrast_model"
# for model in $models
# do
#     echo $model
#     cp -r $model/assets $model/assets_$last_github_commit_ID
# done

# Docker commands:
    # 1)sudo docker run --mount type=bind,source="$(pwd)",target=/mnt --gpus all -it --rm tensorflow/tensorflow:1.7.0-gpu-py3 bash
    # 2)pip install keras==2.2.0 tensorflow_hub==0.1.1 tqdm lime
    # 1)Commit a docker container (with all installed libraries) with a new tag
    # 2)Push the commit on docker hub

# Singularity command:
    # 1)singularity pull docker://sunbro/image_name:tag
    # 2)singularity shell -B /scratch --nv image_name.sif

# SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity shell -B /scratch --nv tar_file.sif

