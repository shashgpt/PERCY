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
    # 0)a)docker login -u sunbro
    #   b)sudo docker pull tensorflow/tensorflow:2.7.1-gpu-jupyter
    # 1)sudo docker run --mount type=bind,source="$(pwd)",target=/mnt --gpus all -it --rm sunbro/compling2022:tensorflow bash
    # 2)code and test your programme
    # 3)commit a docker container (with all installed libraries): 1)sudo docker container ls 2)docker commit <container_ID> sunbro/compling2022:tensorflow
    # 4)change the tag: docker image tag tensorflow/tensorflow:2.7.1-gpu-jupyter sunbro/compling2022:tensorflow
    # 5)push the commit on docker hub: docker image push sunbro/compling2022:tensorflow
    # 6)save the dockerfile as file.tar to use on cedar

# Singularity commands:
    # 1)singularity pull docker://sunbro/compling2022:tensorflow (DON'T DO THIS)
    # 2)save the docker image as image_name.tar (docker save sunbro/compling2022:tensorflow > compling2022_tensorflow.tar) (DO THIS)
    # 3)upload the compling2022_tensorflow.tar file on cedar (sftp to cedar, upload in scratch folder)
    # 4)build from tar file (singularity build compling2022_tensorflow.sif docker-archive://compling2022_tensorflow.tar)
    # 5)SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity shell -B /scratch --nv compling2022_tensorflow.sif

# Gcloud commands:
    # 1) gcloud compute config-ssh
    # 2) source ~/tensorflow/bin/activate
    # 3) source ~/pytorch/bin/activate

# Send the code on cedar:
    # rsync -a -e "ssh -i ~/.ssh/id_rsa" --relative "$PWD"/./  --exclude assets*/ --exclude datasets*/ --exclude analysis*/ --exclude git_commands.sh --exclude run.sh --exclude requirements.txt reda@206.12.124.2:/scratch/reda/shashank/CompLing2022/${PWD##*/}

# Send the code on pippin:
    # sshpass -p "#Deakin2630" rsync -a --relative "$PWD"/./ --exclude assets*/ --exclude datasets*/ --exclude results*/ --exclude runs*/ --exclude stanford-corenlp-full-2018-10-05/parser_datasets*/ guptashas@pippin.it.deakin.edu.au:"$PWD"/

# Send the code on google drive:
    # rsync -avr --exclude datasets*/ --exclude assets*/ --exclude analysis*/ --exclude git_commands.sh --exclude run.sh --exclude requirements.txt $PWD /home/guptashas/Google_Drive/PhD_experiments/CompLing2022/${PWD##*/}

# Send the code to GCP VM: (vm1.us-central1-a.sit-shashank-gupta)
    # rsync -avr --relative "$PWD"/./ --exclude assets*/ --exclude datasets*/ --exclude stanford-corenlp-full-2018-10-05/parser_datasets*/ --exclude stanford-corenlp-full-2018-10-05/parse_trees*/ --exclude runs*/ --exclude results*/ vm1.us-central1-a.sit-shashank-gupta:"$PWD"/
    # rsync -avr --relative "$PWD"/./ --exclude datasets*/ --exclude analysis*/ --exclude git_commands.sh --exclude run.sh --exclude requirements.txt vm1.us-central1-a.sit-shashank-gupta:"$PWD"/


# Receive assets from cedar:
    # rsync -a -e "ssh -i ~/.ssh/id_rsa" -a --exclude assets/preprocessed_dataset*/ --exclude assets/word_index*/ --exclude assets/word_vectors*/ --exclude scripts*/ --exclude main* --exclude datasets*/ --exclude analysis*/ --exclude config.py --exclude git_commands.sh --exclude main.py --exclude run.sh --exclude requirements.txt reda@206.12.124.2:/scratch/reda/shashank/CompLing2022/${PWD##*/} "$PWD"

# Receive assets from pippin:
    # sshpass -p "#Deakin2630" rsync -a --exclude scripts*/ --exclude main* --exclude datasets*/ --exclude analysis*/ --exclude config.py --exclude git_commands.sh --exclude main.py --exclude run.sh --exclude requirements.txt guptashas@pippin.it.deakin.edu.au:"$PWD"/ "$PWD"/
    # ssh -i ~/.ssh/id_rsa guptashas@luthin.it.deakin.edu.au "rm -rf "$PWD"/assets/"

# Receive assets from GCP:
    # rsync -avr --exclude scripts*/ --exclude main* --exclude datasets*/ --exclude analysis*/ --exclude config.py --exclude git_commands.sh --exclude main.py --exclude run.sh --exclude requirements.txt vm1.us-central1-a.sit-shashank-gupta:"$PWD"/ "$PWD"/
    # rsync -avr --include="assets*/" --include="results*/" --exclude="*" vm1.us-central1-a.sit-shashank-gupta:"$PWD"/ "$PWD"/

# Cedar request jobs
    # sbatch  --time=240:00:0 --mem=64G --cpus-per-task=2 --ntasks=1 --gres=gpu:4 --account=rrg-ssanner --mail-user=shashankgupta314@gmail.com  --mail-type=ALL   ~/wait.sh

# Cedar jupyter-notebook port-forwarding
    # ssh -i ~/.ssh/id_rsa -N -f -L localhost:9509:localhost:9509 reda@206.12.124.2