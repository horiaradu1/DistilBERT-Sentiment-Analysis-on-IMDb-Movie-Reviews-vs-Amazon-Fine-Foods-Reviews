#!/bin/bash

# This script will ensure the starting files are up-to-date in the current branch 
# 
# It does this by fetching the base repository (the one we control) and merging 
# with it. If you have already modified these starting files then you may need
# to resolve some conflicts. 
#
# You should run this script at the start of an exercise and if told to do so. 
#
# This script can also be used to refresh a single file e.g.
# ./refresh.sh refresh.sh
# will update this script
#
# Author: Giles Reger

# find tag name
BRANCH=$(git branch | grep "\*" | cut -d ' ' -f 2)

# In case base was previously added wrongly, just delete and re-add 
git remote remove base
git remote add base https://gitlab.cs.man.ac.uk/z07959vs/comp34812_2021_base.git

# Get any base changes
git fetch base

# If no file given just merge full branch 
if [ -z $1 ]; then
  git merge base/"${BRANCH}"
else
  git checkout base/"${BRANCH}" $1
fi
