#!/usr/bin/bash

# Series of commands to either push or pull from github repo
if [ $1 == "push" ]
then
    git init
    git rm --cached . -r
    git add --all -- ':!bert/datasets/*' ':!bert/assets/*' ':!elmo_pytorch/datasets/*' ':!elmo_pytorch/assets/*' ':!w2v_and_glove/datasets/*' ':!w2v_and_glove/assets/*' ':!git_commands.sh/*' ':!run.sh/*' ':!senti_bert/assets/*' ':!senti_bert/datasets/*' ':!senti_bert/results/*' ':!senti_bert/runs/*' ':!senti_bert/stanford-corenlp-full-2018-10-05/*'
    git status
    git commit -m "Commit message"
    git branch -M main
    git remote add origin "https://github.com/shashank0117/CompLing2022.git"
    git remote -v
    git push origin main
    git log --oneline # get the commit ID
elif [ $1 == "pull" ]
then
    git pull origin main
fi

