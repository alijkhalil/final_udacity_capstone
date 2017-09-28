#!/bin/sh

UTIL_REPO_NAME="dl_utilities"
CNN_MODEL_REPO_NAME="state_of_art_cnns"

cd `dirname $0`

# Get necessary repo's
echo -n "Getting other repositories needed for the project...  "

if [ ! -d $UTIL_REPO_NAME ]; then
    git clone git@github.com:alijkhalil/"$UTIL_REPO_NAME".git > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Download error!"
        exit 1
    fi
fi

if [ ! -d $CNN_MODEL_REPO_NAME ]; then
    git clone git@github.com:alijkhalil/"$CNN_MODEL_REPO_NAME".git > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Download error!"
        exit 1
    fi
fi

# Print success and exit
echo "Done!"
exit 0