source env.sh
parsecmgmt -a build -p libtool
parsecmgmt -a build -p hooks
parsecmgmt -a build -p blackscholes -c gcc-hooks
parsecmgmt -a build -p bodytrack -c gcc-hooks
parsecmgmt -a build -p canneal -c gcc-hooks
parsecmgmt -a build -p dedup -c gcc-hooks
parsecmgmt -a build -p facesim -c gcc-hooks
parsecmgmt -a build -p ferret -c gcc-hooks
parsecmgmt -a build -p fluidanimate -c gcc-hooks
parsecmgmt -a build -p freqmine -c gcc-hooks
parsecmgmt -a build -p streamcluster -c gcc-hooks
parsecmgmt -a build -p swaptions -c gcc-hooks
parsecmgmt -a build -p vips -c gcc-hooks
parsecmgmt -a build -p x264 -c gcc-hooks
echo "12345" | sudo -S chown gem5 -R /usr/local/
echo "12345" | sudo -S chgrp gem5 -R /usr/local/
parsecmgmt -a build -p raytrace -c gcc-hooks
cp -r /usr/local/bin/ /home/gem5/parsec-benchmark/pkgs/tools/cmake/inst/amd64-linux.gcc-hooks/
parsecmgmt -a build -p raytrace -c gcc-hooks
cp -r /usr/local/bin/ /home/gem5/parsec-benchmark/pkgs/apps/raytrace/inst/amd64-linux.gcc-hooks/
echo "12345" | sudo -S chown root -R /usr/local/
echo "12345" | sudo -S chgrp root -R /usr/local/
./get-inputs
