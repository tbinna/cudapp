/*
 * device_array2d.h
 *
 *  Created on: Aug 7, 2012
 *      Author: tbinna
 */

#ifndef DEVICE_ARRAY2D_H_
#define DEVICE_ARRAY2D_H_

#include "array2d.h"
#include "device_allocator.h"
#include "device_memcpy_policy.h"

namespace cudapp {

template<typename T, typename Alloc = cudapp::device_allocator<T>, typename CopyPolicy = cudapp::device_memcpy_policy<T> >
class device_array2d : public cudapp::array2d<T,Alloc,CopyPolicy> {
private:
	typedef cudapp::array2d<T,Alloc,CopyPolicy> Parent;

public:

	__host__
	explicit device_array2d(size_t dimX, size_t dimY, const T& x = T()) : Parent(dimX, dimY, x) {}

	__host__
	explicit device_array2d(const device_array2d<T,Alloc,CopyPolicy> &other) : Parent(other) {}

	__host__
	explicit device_array2d(device_array2d<T,Alloc,CopyPolicy> &other) : Parent(other) {}

	template<typename OtherT, typename OtherAlloc, typename OtherCopyPolicy>
	__host__
	device_array2d(const array2d<OtherT,OtherAlloc,OtherCopyPolicy> &other) : Parent(other) {}
};

}


#endif /* DEVICE_ARRAY2D_H_ */
