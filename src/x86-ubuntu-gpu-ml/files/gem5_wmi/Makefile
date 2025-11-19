obj-m += gem5_wmi.o

all:
	make -C /lib/modules/`uname -r`/build M=${PWD}
clean:
	make -C /lib/modules/`uname -r`/build M=${PWD} clean
