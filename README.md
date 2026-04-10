# Robotics
This repository contains all the code for the mobile robots challenge and the final project for robotics

# --------------- IMPORTANT ---------------

*Create own branch first* 

In order to push files to the remote repository you need you own branch. You can only change the main branch through a merge request. To create your own branch run in your terminal the following 
command: 

`git branch "your-branch-name"`

Change `"your-branch-name"` to how you want to name your branch.
This will create the new branch, with the name that you specified under `"your-branch-name"`, for you to work on. To switch to that branch you do the following:

`git checkout "your-branch-name"`

This will switch you to the branch you've just created. However, the branch is only visible on your computer, to make it visible to everybody else you must publish your branch with the last 
command:

`git push -u origin "your-branch-name"`

Now everything should be set up. In the future make shure that you are working on your branch. You can check that by running `git brach` in your terminal. This will display all the branches that 
are present in the repository. If your branch is displayed as `* "your-branch-name"` than you are good to go! 

# Requirements 

We recommend using a conda environment:

`conda create -n "env-name" python=3.11` <!-- You need python=3.11 for the library pyaudio -->

`conda install numpy matplotlib` 
<!-- `conda install -c conda-forge librosa` -->
`conda install -c main::pyaudio`
