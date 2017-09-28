#!/bin/sh

# Set up variables and local directory
UTIL_REPO_NAME="dl_utilities"
UTIL_REPO_GIT_HASH="a4c70c3c35794a85c411ffda5bd5809027d310f1"

CNN_MODEL_REPO_NAME="state_of_art_cnns"
CNN_MODEL_REPO_GIT_HASH="fa32da3967572f37de399241a87bcc9f7a9e028a"

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

cd $UTIL_REPO_NAME
git checkout $UTIL_REPO_GIT_HASH > /dev/null 2>&1
cd - > /dev/null 2>&1

if [ ! -d $CNN_MODEL_REPO_NAME ]; then
    git clone git@github.com:alijkhalil/"$CNN_MODEL_REPO_NAME".git > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Download error!"
        exit 1
    fi
fi

cd $CNN_MODEL_REPO_NAME
git checkout $CNN_MODEL_REPO_GIT_HASH > /dev/null 2>&1
cd - > /dev/null 2>&1


# Print success and exit
echo "Done!"
exit 0