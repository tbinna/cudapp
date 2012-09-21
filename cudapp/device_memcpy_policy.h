/*
 * device_memcpy_policy.h
 *
 *  Created on: Aug 7, 2012
 *      Author: tbinna
 */

#ifndef DEVICE_MEMCPY_POLICY_H_
#define DEVICE_MEMCPY_POLICY_H_

namespace cudapp {

template<typename T>
class device_memcpy_policy {
public:
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // memcopy in the same address space device -> device
    void copy_same(pointer dst, const_pointer src, size_type cnt) {
		cudaError_t error = cudaMemcpy(dst, src, cnt * sizeof(T), cudaMemcpyDeviceToDevice);
		if(error)
			throw cudapp::cuda_error(error, "memcopy device->device failed.");
    }

    // memcopy to the other address space host -> device
    void copy_other(pointer dst, const_pointer src, size_type cnt) {
		cudaError_t error = cudaMemcpy(dst, src, cnt * sizeof(T), cudaMemcpyHostToDevice);
		if(error)
			throw cudapp::cuda_error(error, "memcopy host->device failed.");
    }

};

}


#endif /* DEVICE_MEMCPY_POLICY_H_ */
