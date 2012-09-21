#include "cute.h"
#include "ide_listener.h"
#include "cute_runner.h"
#include "array2dTest.cuh"

#include "host_array2d.h"
#include "device_array2d.h"

template<typename DeviceOp>
__global__
void run_device_kernel(DeviceOp op) {
	op();
}

template<typename DeviceOp>
void test_device(DeviceOp op, dim3 gDim = 1, dim3 bDim = 1) {
	run_device_kernel<<<gDim,bDim>>>(op);
}

// host side tests

void testInit() {
	cudapp::host_array2d<int> a(10, 10);
	ASSERT_EQUAL(100, a.size);
	ASSERT_EQUAL(10, a.dimX);
	ASSERT_EQUAL(10, a.dimY);

	for(int i = 0; i < a.size; i++) {
		ASSERT_EQUAL(0, *a[i]);
	}
}

void testInitZero() {
	ASSERTM("Implement this test and code.", false);
	//cudapp::array2d<int> a(1,0);
	//cudapp::array2d<int> a(0,1);
	//cudapp::array2d<int> a(0,0);
}

void testSet() {
	cudapp::host_array2d<int> a(10, 10);
	a.set(0, 0, 10);
	a.set(5,4);
	ASSERT_EQUAL(10, a.get(0,0));
	ASSERT_EQUAL(1, a.get(5,4));
}

void testConstructionCtor(){
	cudapp::host_array2d<int> a(10, 10);

	cudapp::device_array2d<int> b = a;
	ASSERT_EQUAL(100, b.size);
	ASSERT_EQUAL(10, b.dimX);
	ASSERT_EQUAL(10, b.dimY);
}

// device test

struct device_inc {
	device_inc(cudapp::device_array2d<int> &a) : a(a) {}

	__device__
	void operator()() {
		a.inc(5, 5, 10);
	}

	cudapp::device_array2d<int> a;
};

void testDeviceInc(){
	cudapp::host_array2d<int> a(10, 10);
	ASSERT_EQUAL(0, a.get(5,5));
	cudapp::device_array2d<int> b = a;
	test_device(device_inc(b));
	a = b;
	ASSERT_EQUAL(10, a.get(5,5));
}


cute::suite make_suite_array2d() {
	cute::suite s;
	s.push_back(CUTE(testSet));
	s.push_back(CUTE(testInit));
	s.push_back(CUTE(testInitZero));
	s.push_back(CUTE(testConstructionCtor));
	s.push_back(CUTE(testDeviceInc));
	return s;
}

