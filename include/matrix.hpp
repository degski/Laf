
// MIT License
//
// Copyright (c) 2019 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include <algorithm>

#include <sax/iostream.hpp> // <iostream> + nl, sp etc. defined...
#include <random> // *
#include <string>
#include <type_traits> // *
#include <utility> // *

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_trans.h>

#include <sax/autotimer.hpp>
#include <sax/integer.hpp>
#include <sax/prng.hpp>


#ifdef USE_TBB_ALLOCATOR
#include <tbb/scalable_allocator.h>
#else
#ifndef scalable_aligned_malloc
#define scalable_aligned_malloc mkl_malloc
#endif
#ifndef scalable_aligned_free
#define scalable_aligned_free mkl_free
#endif
#endif


#include "serialize.hpp"
#include "type_traits.hpp"


enum class matrix_creation { none = 0, zero, fill, uniform, gaussian };


template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
struct matrix { // 16 or 24 bytes

    using matrix_ptr = matrix*;
    using reference = real&;
    using pointer = real*;
    using properties = std::uintptr_t;
    using value_type = real;
    using generator = sax::Rng;

    pointer m_data = nullptr; // 8 bytes.
    index rows = 0, cols = 0; // 8 or 16 bytes.

    matrix ( ) { }
    matrix ( pointer ptr_, const index rows_, const index cols_, matrix_creation ct_ = matrix_creation::none, const real lb_ = real { -1 }, const real ub_ = real { 1 } );
    matrix ( const index rows_, const index cols_, matrix_creation ct_ = matrix_creation::none, const real lb_ = real { -1 }, const real ub_ = real { 1 } );
    matrix ( const std::pair<index, index> & dims_, matrix_creation ct_ = matrix_creation::none, const real lb_ = real { -1 }, const real ub_ = real { 1 } );
    ~matrix ( );

    pointer data ( ) const noexcept { return untag ( m_data ); }

    // Properties...

    bool is_square ( ) const noexcept { return is_square ( m_data ); }
    bool is_not_square ( ) const noexcept { return not ( is_square ( m_data ) ); }
    bool is_assigned ( ) const noexcept { return not ( is_not_assigned ( m_data ) ); }
    bool is_not_assigned ( ) const noexcept { return is_not_assigned ( m_data ); }

    // sizes...

    index elem ( ) const noexcept { return rows * cols; }
    static constexpr index matrix_size ( ) noexcept { return index { 8 } * ( index ) sizeof ( void* ); } // sax::nextPowerOfTwo ( sizeof ( matrix ) );
    static constexpr index alignment ( ) noexcept { return index { 8 } * ( index ) sizeof ( void* ); }
    index data_size ( ) const noexcept { return rows * cols * ( index ) sizeof ( real ); }

    // Linear algebra...

    std::pair<matrix_ptr, lapack_int> invert ( ) const;
    std::pair<matrix_ptr, lapack_int> solve ( const matrix_ptr b_ ) const;

    // Values...

    real trace ( ) const noexcept;
    std::pair<value_type, value_type> min_max ( ) const noexcept;
    reference at ( const index r_, const index c_ ) noexcept;
    value_type at ( const index r_, const index c_ ) const noexcept;

    // Copy...

    void transpose ( );
    matrix_ptr construct_copy ( ) const;
    matrix_ptr construct_transpose ( ) const;
    void fill ( const real v_ );

    // Hash...

    index hash ( ) const noexcept;

    // Static methods...

    static matrix_ptr zero ( const matrix_ptr m_ );
    static matrix_ptr fill ( const matrix_ptr m_, const real v_ );
    static matrix_ptr gaussian ( const matrix_ptr m_ );
    static matrix_ptr uniform ( const matrix_ptr m_, const real lb_ = real { -1 }, const real ub_ = real { 1 } );

    static matrix_ptr construct ( const index rows_, const index cols_ );
    static matrix_ptr construct_zero ( const index rows_, const index cols_ );
    static matrix_ptr construct_fill ( const index rows_, const index cols_, const real v_ );
    static matrix_ptr construct_gaussian ( const index rows_, const index cols_ );
    static matrix_ptr construct_uniform ( const index rows_, const index cols_, const real lb_, const real ub_ );

    static void destroy ( matrix_ptr & ptr_ ) noexcept;

    static void seed ( ) noexcept { rng.seed ( seed_rng ( ) ); }
    static void seed ( const generator::result_type v_ ) noexcept { rng.seed ( v_ ); }

    template<typename T, typename S = std::size_t>
    class pointer_iterator {

        public:

        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using size_type = S;
        using reference = T&;
        using pointer = T*;
        using iterator_category = std::bidirectional_iterator_tag;

        explicit pointer_iterator ( const pointer & p_ ) noexcept : m_pointer ( p_ ) { }
        explicit pointer_iterator ( pointer && p_ ) noexcept : m_pointer ( std::move ( p_ ) ) { }

        pointer_iterator ( const pointer_iterator & pi_ ) noexcept : m_pointer ( pi_.m_pointer ) { }
        pointer_iterator ( pointer_iterator && pi_ ) noexcept : m_pointer ( std::move ( pi_.m_pointer ) ) { }

        pointer_iterator & operator ++ ( ) noexcept { ++m_pointer; return *this; }
        pointer_iterator operator ++ ( int ) noexcept { pointer_iterator it ( *this ); ++( *this ); return it; }

        pointer_iterator & operator -- ( ) noexcept { --m_pointer; return *this; }
        pointer_iterator operator -- ( int ) noexcept { pointer_iterator it ( *this ); --( *this ); return it; }

        bool operator == ( pointer_iterator & other_ ) const noexcept { return m_pointer == other_.m_pointer; }
        bool operator != ( pointer_iterator & other_ ) const noexcept { return not ( *this == other_ ); }

        pointer_iterator operator + ( const size_type i_ ) const noexcept { return pointer_iterator { m_pointer + i_ }; }
        pointer_iterator operator - ( const size_type i_ ) const noexcept { return pointer_iterator { m_pointer - i_ }; }

        value_type & operator * ( ) const noexcept { return *m_pointer; }

        pointer get ( ) const noexcept { return m_pointer; }

        private:

        pointer m_pointer;
    };

    private:

    friend class cereal::access;

    template<class Archive>
    void save ( Archive & ar_ ) const;
    template<class Archive>
    void load ( Archive & ar_ );

    static pointer tag_is_not_assigned ( pointer p_ ) noexcept { return ( pointer ) ( ( std::uintptr_t ) p_ | std::uintptr_t { 1 } ); }
    static pointer tag_is_square ( pointer p_ ) noexcept { return ( pointer ) ( ( std::uintptr_t ) p_ | std::uintptr_t { 2 } ); }

    static pointer untag ( pointer p_ ) noexcept { return ( pointer ) ( ( std::uintptr_t ) p_ & ~std::uintptr_t { 3 } ); }

    static bool is_not_assigned ( pointer p_ ) noexcept { return ( std::uintptr_t ) p_ & std::uintptr_t { 1 }; }
    static bool is_square ( pointer p_ ) noexcept { return ( std::uintptr_t ) p_ & std::uintptr_t { 2 }; }

    static auto seed_rng ( ) noexcept;
    static constexpr index hash_multiplier ( ) noexcept { if constexpr ( sizeof ( index ) == 4 ) return 0x45D9F3B; else return 0x0CF3FD1B9997F637; }

    static generator rng;
};

template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::generator matrix<real, index, sfinae>::rng ( 123 ); // matrix<real, index, sfinae>::seed_rng ( ) );

#include "matrix.inl"

// Output (to std::cout only)...

#include "matrix_print.inl"

template<typename S, typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
S & operator << ( S & out_, const matrix<real, index, sfinae> & m_ ) {

    detail::print ( m_, index { 3 } );

    return out_;
}


template<typename S, typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
S & operator << ( S & out_, const matrix<real, index, sfinae> * m_ ) {

    detail::print ( *m_, index { 3 } );

    return out_;
}
