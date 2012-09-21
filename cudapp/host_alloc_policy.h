/*
 * host_alloc.h
 *
 *  Created on: Aug 6, 2012
 *      Author: tbinna
 */

#ifndef HOST_ALLOC_POLICY_H_
#define HOST_ALLOC_POLICY_H_

#include <limits>
#include "alloc_policy.h"
#include "cudapp_exception.h"

namespace cudapp {

template<typename T>
class host_alloc_policy : public cudapp::alloc_policy<T> {
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
        typedef host_alloc_policy<U> other;
    };

    explicit host_alloc_policy() {}
    explicit host_alloc_policy(host_alloc_policy const &other) {}

    template <typename U>
    explicit host_alloc_policy(host_alloc_policy<U> const &other) {}

    ~host_alloc_policy() {}

    size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    pointer allocate(size_type cnt, const_pointer = 0) {
    	if (cnt > max_size()) {
    		throw cudapp::invalid_argument("Specified number of elements for allocation exceeds numerical limits.");
    	}

		pointer result = 0;
		result = (pointer) malloc(cnt * sizeof(T));

		if (result == 0) {
			throw cudapp::bad_alloc("Error while allocating host memory.");
		}

		return result;
    }

    void deallocate(pointer p, size_type) throw() {
    	free(p);
    }
};

// determines if memory from another
// allocator can be deallocated from this one
template<typename T, typename T2>
bool operator==(host_alloc_policy<T> const&, host_alloc_policy<T2> const&) {
    return true;
}

template<typename T, typename OtherAllocator>
bool operator==(host_alloc_policy<T> const&, OtherAllocator const&) {
    return false;
}

}

#endif /* HOST_ALLOC_POLICY_H_ */
