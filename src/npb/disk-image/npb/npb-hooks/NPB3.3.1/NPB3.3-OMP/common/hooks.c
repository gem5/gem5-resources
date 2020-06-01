#include <stdio.h>
#include "m5_mmap.h"


void init() __attribute__((constructor));

void init() {

	//__attribute__ makes this function get called before main()
	// need to mmap /dev/mem
	map_m5_mem();
}

void roi_begin_(){

	printf(" -------------------- ROI BEGIN -------------------- \n");
	m5_work_begin(0,0);
	}

void roi_end_(){
       	printf(" -------------------- ROI END -------------------- \n");
	m5_work_end(0,0);
	}
