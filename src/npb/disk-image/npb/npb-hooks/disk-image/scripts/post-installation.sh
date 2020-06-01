echo 'Post Installation Started';
echo '12345' | sudo apt-get install gfortran;
echo 'Building NPB';
cd ~/NPB3.3.1/NPB3.3-OMP/;
mkdir bin;
make suite -j 8;
echo 'Building Done'
echo 'Post Installation Done';
