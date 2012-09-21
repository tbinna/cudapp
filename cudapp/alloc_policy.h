/*
 * alloc_policy.h
 *
 *  Created on: Aug 3, 2012
 *      Author: tbinna
 *
 * The allocation policy determines how the memory allocation/deallocation works,
 * the maximum number of objects of type T that can be allocated, as well as the
 * equality checks to determine if other allocators can allocate and deallocate
 * between allocators interchangeably.
 *
 */

#ifndef ALLOC_POLICY_H_
#define ALLOC_POLICY_H_

#include <limits>

namespace cudapp {

template<typename T>
class alloc_policy {
public:
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // convert an alloc_policy<T> to alloc_policy<U>
//    template<typename U>
//    struct rebind {
//        typedef alloc_policy<U> other;
//    };

    explicit alloc_policy() {}
    explicit alloc_policy(alloc_policy const&) {}

    template <typename U>
    explicit alloc_policy(alloc_policy<U> const&) {}

    ~alloc_policy() {}

//    // memory allocation, define methods allocate and deallocate in derived classes
//    inline pointer allocate(size_type cnt, const_pointer = 0) {
//        return reinterpret_cast<pointer>(::operator new(cnt * sizeof (T)));
//    }
//
//    inline void deallocate(pointer p, size_type) {
//    	::operator delete(p);
//    }


};

// determines if memory from another
// allocator can be deallocated from this one
//template<typename T, typename T2>
//inline bool operator==(alloc_policy<T> const&, alloc_policy<T2> const&) {
//    return true;
//}
//
//template<typename T, typename OtherAllocator>
//inline bool operator==(alloc_policy<T> const&, OtherAllocator const&) {
//    return false;
//}

}

#endif /* ALLOC_POLICY_H_ */
