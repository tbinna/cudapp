/*
 * array2d.h
 *
 *  Created on: Aug 3, 2012
 *      Author: tbinna
 */

#ifndef ARRAY2D_H_
#define ARRAY2D_H_

#include "cudapp_exception.h"

namespace cudapp {

template<typename T, typename Alloc, typename CopyPolicy>
class array2d : public CopyPolicy {
public:
	typedef typename Alloc::size_type size_type;
	typedef typename Alloc::difference_type difference_type;
    typedef typename Alloc::pointer pointer;
    typedef typename Alloc::const_pointer const_pointer;
	typedef typename Alloc::reference reference;
	typedef typename Alloc::const_reference const_reference;
	typedef typename Alloc::value_type value_type;

	size_type dimX;
	size_type dimY;
	size_type size;

	__host__
	explicit array2d(size_t dimX, size_t dimY, const T& x = T()) : dimX(dimX), dimY(dimY), size(dimX*dimY), alloc() {
		if (size != 0) {
			data = _allocate(size);
			_uninitialized_fill(data, size, x);
			__sync_fetch_and_add(&refcount, 1);
		}
	}

	__host__
	explicit array2d(const array2d<T,Alloc,CopyPolicy> &other) : dimX(other.dimX), dimY(other.dimY), size(other.size), alloc(other.alloc), data(other.data) {
		__sync_fetch_and_add(&refcount, 1); // TODO: must be atomic, this way only works with gcc
	}

	__host__
	explicit array2d(array2d<T,Alloc,CopyPolicy> &other) : dimX(other.dimX), dimY(other.dimY), size(other.size), alloc(other.alloc), data(other.data) {
		__sync_fetch_and_add(&refcount, 1); // TODO: must be atomic, this way only works with gcc
	}

	template<typename OtherT, typename OtherAlloc, typename OtherCopyPolicy>
	__host__
	array2d(const array2d<OtherT,OtherAlloc,OtherCopyPolicy> &other) : dimX(other.dimX), dimY(other.dimY), alloc(), size(other.size)  {
		if (size != 0) {
			data = _allocate(size);
			copy_other(data, other[0], size);
			__sync_fetch_and_add(&refcount, 1);
		}
	}

	__host__
	~array2d() {
		__sync_fetch_and_sub(&refcount, 1); //TODO: must be atomic, this way only works with gcc

		if(refcount == 0) {
			_release();
		}
	}

	// container operations

	__host__ __device__
	T get(size_t x, size_t y) const {
		return data[x + y * dimX];
	}

	__host__ __device__
	void set(size_t x, size_t y, T value = 1) {
		data[x+y*dimX] = value;
	}

	__device__
	void inc(size_t x, size_t y, T value = 1) {
		atomicAdd(&data[x+y*dimX], value);
	}

	array2d<T,Alloc,CopyPolicy>& operator=(const array2d<T,Alloc,CopyPolicy> &other) {
		// device to device or host to host
		assert(size == other.size);
		copy_same(data, other[0], size);
		return *this;
	}

	template<typename OtherT, typename OtherAlloc, typename OtherCopyPolicy>
	array2d<T,Alloc,CopyPolicy>& operator=(const array2d<OtherT,OtherAlloc, OtherCopyPolicy> &other) {
		// device to host or host to device
		assert(size == other.size);
		copy_other(data, other[0], size);
		return *this;
	}

	__host__ __device__
	T* operator[](size_t i) {
		return data + i;
	}

	__host__ __device__
	const T* operator[](size_t i) const {
		return data + i;
	}

private:
	Alloc alloc;
	static int refcount;
	typename Alloc::pointer data;

	__host__
	pointer _allocate(size_type n) {
		return n != 0 ? alloc.allocate(n) : 0;
	}

	__host__
	void _release() {
		for (size_type i = 0; i < size; ++i)
			alloc.destroy(data + i);
		alloc.deallocate(data, size);
	}

	__host__
	void _uninitialized_fill(pointer p, size_type size, const_reference x) {
		size_t i;
		try {
			for (i = 0; i < size; ++i) {
				alloc.construct(p + i, x);
			}
		} catch (...) { // in case of an error do clean up
			for (size_type j = 0; j < i; ++j) {
				alloc.destroy(p + j);
			}
			alloc.deallocate(p, size);
			throw cudapp::bad_init("Failed to construct values.");
		}
	}
};

template<typename T, typename Alloc, typename CopyPolicy>
int array2d<T,Alloc,CopyPolicy>::refcount = 0;

}

#endif /* ARRAY2D_H_ */
