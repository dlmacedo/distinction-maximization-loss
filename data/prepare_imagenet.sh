#!/bin/bash

#!/bin/bash

echo "Begining..."

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
rm imagenet-o.tar

echo "Done!!!"