/*
 * host_memcpy_policy.h
 *
 *  Created on: Aug 7, 2012
 *      Author: tbinna
 */

#ifndef HOST_MEMCPY_POLICY_H_
#define HOST_MEMCPY_POLICY_H_

namespace cudapp {

template<typename T>
class host_memcpy_policy {
public:
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // memcopy in the same address space host -> host
    void copy_same(pointer dst, const_pointer src, size_type cnt) {
    	memcpy(dst, src, cnt * sizeof(T));
    }

    // memcopy to the other address space device -> host
    void copy_other(pointer dst, const_pointer src, size_type cnt) {
		cudaError_t error = cudaMemcpy(dst, src, cnt * sizeof(T), cudaMemcpyDeviceToHost);
		if(error)
			throw cudapp::cuda_error(error, "memcopy device->host failed.");
    }
};

}

#endif /* HOST_MEMCPY_POLICY_H_ */
