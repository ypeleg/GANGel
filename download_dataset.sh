URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip
ZIP_FILE=./apple2orange.zip
TARGET_DIR=./apple2orange/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./
rm $ZIP_FILE