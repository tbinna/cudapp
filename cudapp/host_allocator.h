/*
 * host_allocator.h
 *
 *  Created on: Aug 6, 2012
 *      Author: tbinna
 */

#ifndef HOST_ALLOCATOR_H_
#define HOST_ALLOCATOR_H_

#include "allocator_base.h"
#include "host_alloc_policy.h"

namespace cudapp {

template<typename T>
class host_allocator : public cudapp::allocator_base<T, cudapp::host_alloc_policy<T> > {
private:
	typedef cudapp::allocator_base<T, cudapp::host_alloc_policy<T> > Parent;

public:
    typedef typename Parent::size_type size_type;
    typedef typename Parent::difference_type difference_type;
    typedef typename Parent::pointer pointer;
    typedef typename Parent::const_pointer const_pointer;
    typedef typename Parent::reference reference;
    typedef typename Parent::const_reference const_reference;
    typedef typename Parent::value_type value_type;

	explicit host_allocator() : Parent() {}
	host_allocator(host_allocator const& rhs) : Parent(rhs) {}

    template<typename U>
    host_allocator(host_allocator<U> const& rhs) : Parent(rhs) {}

    ~host_allocator() {}

};

}

#endif /* HOST_ALLOCATOR_H_ */
