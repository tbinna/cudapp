/*
 * allocator_base.h
 *
 *  Created on: Aug 3, 2012
 *      Author: tbinna
 *
 *  Based on the tutorial on:
 *  http://www.codeproject.com/Articles/4795/C-Standard-Allocator-An-Introduction-and-Implement
 */

#ifndef ALLOCATOR_BASE_H_
#define ALLOCATOR_BASE_H_

#include "object_traits.h"

namespace cudapp {

template<typename T, typename Policy, typename Traits = cudapp::object_traits<T> >
class allocator_base : public Policy, public Traits {
private:
    typedef Policy AllocationPolicy;
    typedef Traits TTraits;

public:
    typedef typename AllocationPolicy::size_type size_type;
    typedef typename AllocationPolicy::difference_type difference_type;
    typedef typename AllocationPolicy::pointer pointer;
    typedef typename AllocationPolicy::const_pointer const_pointer;
    typedef typename AllocationPolicy::reference reference;
    typedef typename AllocationPolicy::const_reference const_reference;
    typedef typename AllocationPolicy::value_type value_type;

//    template<typename U>
//    struct rebind {
//        typedef allocator_base<U, typename AllocationPolicy::rebind<U>::other, typename TTraits::rebind<U>::other > other;
//    };

    explicit allocator_base() {}
    allocator_base(allocator_base const& rhs) : Traits(rhs), Policy(rhs) {}

    template<typename U, typename P, typename T2>
    allocator_base(allocator_base<U, P, T2> const& rhs) : Traits(rhs), Policy(rhs) {}

    ~allocator_base() {}
};

// determines if memory from another
// allocator can be deallocated from this one
template<typename T, typename P, typename Tr>
bool operator==(allocator_base<T, P, Tr> const& lhs, allocator_base<T, P, Tr> const& rhs) {
    return operator==(static_cast<P&>(lhs), static_cast<P&>(rhs));
}

template<typename T, typename P, typename Tr, typename T2, typename P2, typename Tr2>
bool operator==(allocator_base<T, P, Tr> const& lhs, allocator_base<T2, P2, Tr2> const& rhs) {
      return operator==(static_cast<P&>(lhs), static_cast<P2&>(rhs));
}

template<typename T, typename P, typename Tr, typename OtherAllocator>
bool operator==(allocator_base<T, P, Tr> const& lhs, OtherAllocator const& rhs) {
    return operator==(static_cast<P&>(lhs), rhs);
}

template<typename T, typename P, typename Tr>
bool operator!=(allocator_base<T, P, Tr> const& lhs, allocator_base<T, P, Tr> const& rhs) {
    return !operator==(lhs, rhs);
}

template<typename T, typename P, typename Tr, typename T2, typename P2, typename Tr2>
bool operator!=(allocator_base<T, P, Tr> const& lhs, allocator_base<T2, P2, Tr2> const& rhs) {
    return !operator==(lhs, rhs);
}

template<typename T, typename P, typename Tr, typename OtherAllocator>
bool operator!=(allocator_base<T, P, Tr> const& lhs, OtherAllocator const& rhs) {
    return !operator==(lhs, rhs);
}

}

#endif /* ALLOCATOR_BASE_H_ */
