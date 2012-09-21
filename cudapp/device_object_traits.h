/*
 * device_object_traits.h
 *
 *  Created on: Aug 7, 2012
 *      Author: tbinna
 */

#ifndef DEVICE_OBJECT_TRAITS_H_
#define DEVICE_OBJECT_TRAITS_H_

namespace cudapp {

template<typename T>
class device_object_traits {
public :
    // convert an object_traits<T> to object_traits<U>
    template<typename U>
    struct rebind {
        typedef device_object_traits<U> other;
    };

    inline explicit device_object_traits() {}
    inline ~device_object_traits() {}

    template <typename U>
    inline explicit device_object_traits(device_object_traits<U> const&) {}

    inline T* address(T& r) { return &r; }
    inline T const* address(T const& r) { return &r; }

    inline void construct(T* p, const T& t) {
    	cudaMemcpy(p, &t, sizeof(T), cudaMemcpyHostToDevice);
    }

    inline void destroy(T* p) {
    	//p->~T(); // TODO: implement, what to do?
    }
};

}


#endif /* DEVICE_OBJECT_TRAITS_H_ */
