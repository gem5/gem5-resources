#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#ifdef FFTW3
#ifdef ESSL_FFTW
#include <fftw3_essl.h>
#else
#include <fftw3.h>
#endif
#else
#include <fftw.h>
#endif

///
// An allocator class based on fftw_malloc to get SIMD friendly
// alignment.
///
template <class T> class fftw_allocator
{
public:
    typedef T                 value_type;
    typedef value_type*       pointer;
    typedef const value_type* const_pointer;
    typedef value_type&       reference;
    typedef const value_type& const_reference;
    typedef std::size_t       size_type;
    typedef std::ptrdiff_t    difference_type;
  
    template <class U> 
    struct rebind { typedef fftw_allocator<U> other; };

    fftw_allocator() {}
    fftw_allocator(const fftw_allocator&) {}
    template <class U> 
    fftw_allocator(const fftw_allocator<U>&) {}
    ~fftw_allocator() {}
    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return x; }

    pointer allocate(size_type n, const_pointer = 0)
        {
            void* p = fftw_malloc(n * sizeof(T));
            if (!p) {
                throw std::bad_alloc();
            }
            return static_cast<pointer>(p);
        }

    void deallocate(pointer p, size_type)
        {
            fftw_free(p);
        }

    size_type max_size() const
        { 
            return static_cast<size_type>(-1) / sizeof(T);
        }

    void construct(pointer p, const value_type& x)
        { 
            new(p) value_type(x); 
        }

    void destroy(pointer p)
        {
            p->~value_type();
        }

private:
    void operator=(const fftw_allocator&);
};


template<> class fftw_allocator<void>
{
    typedef void        value_type;
    typedef void*       pointer;
    typedef const void* const_pointer;

    template <class U> 
    struct rebind { typedef fftw_allocator<U> other; };
};


template <class T>
inline bool operator==(const fftw_allocator<T>&,  const fftw_allocator<T>&)
{
    return true;
}

template <class T>
inline bool operator!=(const fftw_allocator<T>&, 
                       const fftw_allocator<T>&)
{
    return false;
}

#endif
