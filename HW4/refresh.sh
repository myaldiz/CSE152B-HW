# jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace hw3-CSE152B-release.ipynb
# git checkout --orphan newBranch
# git add -A  # Add all files and commit them
git add -u
git commit -m '0'
git branch -D master  # Deletes the master branch
git branch -m master  # Rename the current branch to master
git push -f origin master  # Force push master branch to github
git gc --aggressive --prune=all     # remove the old files