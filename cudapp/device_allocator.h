/*
 * device_allocator.h
 *
 *  Created on: Aug 6, 2012
 *      Author: tbinna
 */

#ifndef DEVICE_ALLOCATOR_H_
#define DEVICE_ALLOCATOR_H_

#include "allocator_base.h"
#include "device_alloc_policy.h"
#include "device_object_traits.h"

namespace cudapp {

template<typename T>
class device_allocator : public cudapp::allocator_base<T, cudapp::device_alloc_policy<T>, cudapp::device_object_traits<T> > {

private:
	typedef cudapp::allocator_base<T, cudapp::device_alloc_policy<T>, cudapp::device_object_traits<T> > Parent;

public:
    typedef typename Parent::size_type size_type;
    typedef typename Parent::difference_type difference_type;
    typedef typename Parent::pointer pointer;
    typedef typename Parent::const_pointer const_pointer;
    typedef typename Parent::reference reference;
    typedef typename Parent::const_reference const_reference;
    typedef typename Parent::value_type value_type;

	explicit device_allocator() : Parent() {}
	device_allocator(device_allocator const& rhs) : Parent(rhs) {}

    template<typename U>
    device_allocator(device_allocator<U> const& rhs) : Parent(rhs) {}

    ~device_allocator() {}


};

}

#endif /* DEVICE_ALLOCATOR_H_ */
