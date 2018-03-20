#!/usr/bin/env bash

# Clean up editor and temp files from the local directory (even if not 
# tracked by git)
echo "Deleting editor temporary files"
find . -name "*.pyc" -exec rm -rf {} \; 2>/dev/null
find . -name "*~" -exec rm -rf {} \;  2>/dev/null

# Add any new files, add all updates to all files

echo "Adding all changes"
git add --all . 
git add -u :/

# Commit using the message specified as first argument to this script

git rm *.swp
rm *.swp
git rm *.swp
git rm *.swp
echo "Git commit"
git commit -m "$1"

# Synchronize with master on github
echo "git pull"
git pull

echo "git push"
git push origin master
