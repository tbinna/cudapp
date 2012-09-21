/*
 * object_traits.h
 *
 *  Created on: Aug 3, 2012
 *      Author: tbinna
 *
 *  With the ObjectTraits, should a creator of T decide to
 *  construct, destroy, or overload operator &, he could do
 *  a complete template specialization of ObjectTraits for
 *  his purpose.
 */

#ifndef OBJECT_TRAITS_H_
#define OBJECT_TRAITS_H_

namespace cudapp {

template<typename T>
class object_traits {
public :
    // convert an object_traits<T> to object_traits<U>
    template<typename U>
    struct rebind {
        typedef object_traits<U> other;
    };

    inline explicit object_traits() {}
    inline ~object_traits() {}

    template <typename U>
    inline explicit object_traits(object_traits<U> const&) {}

    inline T* address(T& r) { return &r; }
    inline T const* address(T const& r) { return &r; }

    inline void construct(T* p, const T& t) { new(p) T(t); }
    inline void destroy(T* p) { p->~T(); }
};

}

#endif /* OBJECT_TRAITS_H_ */
