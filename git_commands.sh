#!/usr/bin/bash

# Series of commands to either push or pull from github repo
if [ $1 == "push" ]
then
    git init
    git rm --cached . -r
    git add --all -- ':!datasets/*' ':!assets/*'
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

