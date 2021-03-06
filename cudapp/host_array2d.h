/*
 * host_array2d.h
 *
 *  Created on: Aug 7, 2012
 *      Author: tbinna
 */

#ifndef HOST_ARRAY2D_H_
#define HOST_ARRAY2D_H_

#include "array2d.h"
#include "host_allocator.h"
#include "host_memcpy_policy.h"

namespace cudapp {

template<typename T, typename Alloc = cudapp::host_allocator<T>,typename CopyPolicy = cudapp::host_memcpy_policy<T> >
class host_array2d : public cudapp::array2d<T,Alloc,CopyPolicy> {
private:
	typedef cudapp::array2d<T,Alloc,CopyPolicy> Parent;

public:

	__host__
	explicit host_array2d(size_t dimX, size_t dimY, const T& x = T()) : Parent(dimX, dimY, x) {}

	__host__
	explicit host_array2d(const host_array2d<T,Alloc,CopyPolicy> &other) : Parent(other) {}

	__host__
	explicit host_array2d(host_array2d<T,Alloc,CopyPolicy> &other) : Parent(other) {}

	template<typename OtherT, typename OtherAlloc, typename OtherCopyPolicy>
	__host__
	host_array2d(const array2d<OtherT, OtherAlloc, OtherCopyPolicy> &other) : Parent(other) {}
};

}


#endif /* HOST_ARRAY2D_H_ */
