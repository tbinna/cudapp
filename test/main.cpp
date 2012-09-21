#include "cute.h"
#include "ide_listener.h"
#include "cute_runner.h"
#include "array2dTest.cuh"

void runSuites(){
	cute::ide_listener lis;
	cute::suite array2d = make_suite_array2d();
	cute::makeRunner(lis)(array2d, "array2d");
}

int main(){
    runSuites();
    return 0;
}



