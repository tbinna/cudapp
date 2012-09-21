/*
 * device_alloc.h
 *
 *  Created on: Aug 3, 2012
 *      Author: tbinna
 */

#ifndef DEVICE_ALLOC_POLICY_H_
#define DEVICE_ALLOC_POLICY_H_

#include <limits>
#include "cudapp_exception.h"

namespace cudapp {

template<typename T>
class device_alloc_policy {
public:
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template<typename U>
    struct rebind {
        typedef device_alloc_policy<U> other;
    };

    explicit device_alloc_policy() {}
    explicit device_alloc_policy(device_alloc_policy const&other) {}

    template <typename U>
    explicit device_alloc_policy(device_alloc_policy<U> const &other) {}

    ~device_alloc_policy() {}

    size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    pointer allocate(size_type cnt, const_pointer = 0) {
    	if (cnt > max_size()) {
    		throw cudapp::invalid_argument("Specified number of elements for allocation exceeds numerical limits.");
    	}

		pointer result = 0;
		cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&result), cnt * sizeof(T));

		if (error) {
			throw cudapp::cuda_error(error);
		}

		return result;
    }

    void deallocate(pointer p, size_type) throw() {
		cudaFree(p); // TODO: is this correct? cuda free can fail. how to handle?
		// do nothing? should be nothrow because called from dtor
    }
};

// determines if memory from another
// allocator can be deallocated from this one
template<typename T, typename T2>
bool operator==(device_alloc_policy<T> const&, device_alloc_policy<T2> const&) {
    return true;
}

template<typename T, typename OtherAllocator>
bool operator==(device_alloc_policy<T> const&, OtherAllocator const&) {
    return false;
}

}

#endif /* DEVICE_ALLOC_POLICY_H_ */
