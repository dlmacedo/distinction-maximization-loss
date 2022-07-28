#!/bin/bash

echo "Begining..."

# download and unzip dataset
#aria2c -x 16 http://cs231n.stanford.edu/tiny-imagenet-200.zip
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
current="$(pwd)/tiny-imagenet-200"
# training data
cd $current/train
for DIR in $(ls); do
   echo "train..."
   cd $DIR
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done
# validation data
cd $current/val
annotate_file="val_annotations.txt"
length=$(cat $annotate_file | wc -l)
for i in $(seq 1 $length); do
    echo "val..."
    # fetch i th line
    line=$(sed -n ${i}p $annotate_file)
    # get file name and directory name
    file=$(echo $line | cut -f1 -d" " )
    directory=$(echo $line | cut -f2 -d" ")
    mkdir -p $directory
    mv images/$file $directory
done
rm -r images
cd ..
cd ..
echo "done"

# download and unzip dataset
#aria2c -x16 https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar
tar -xvf imagenet-o.tar
find . -name "*.txt" -delete
current="$(pwd)/imagenet-o"
cd $current
for DIR in $(ls); do
   echo "loop..."
   cd $DIR
   mv * ..
   cd ..
   rm -rf $DIR
done
cd $current
echo $current
mkdir "$current/imagenet-o"
mv *.JPEG "$current/imagenet-o"
cd ..

cp -rf imagenet-o imagenet-o-64
python resize.py
rm imagenet-o.tar
rm tiny-imagenet-200.zip

echo "Done!!!"
