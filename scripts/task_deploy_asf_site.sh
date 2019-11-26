#!/bin/bash
# Deploy the website to the asf-site branch.
set -e
set -u

echo "Start to generate and deploy site ..."
rm -rf _site
jekyll b --safe
cp .gitignore _site
git checkout asf-site

# remove all existing files
git ls-files | xargs  rm -f
# copy new files into the current site
cp -rf _site/* .
git add --all && git commit -am 'nigthly build at `date`'
git push origin asf-site
git checkout master
echo "Finish deployment..."
