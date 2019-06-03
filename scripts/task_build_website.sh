#!/bin/bash
rm -rf _site
jekyll b --safe
tar czf website.tgz _site
