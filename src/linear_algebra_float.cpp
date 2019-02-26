
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

#include <cfloat> // for FLT_MAX, FLT_MIN

#include <mutex>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <mkl_lapacke.h>

#include <sax/srwlock.hpp>
#include <sax/prng.hpp>
#include <sax/integer.hpp>

#include "linear_algebra_float.h"


#define NOEXCEPT


fmat *linear_algebra_float_malloc ( const std::uint32_t n_rows, const std::uint32_t n_cols ) {

    const std::uint32_t n_elem = n_rows * n_cols;

    if ( n_elem == 0 ) {

        return nullptr;
    }

    const std::uint32_t d_size = n_elem * ( std::uint32_t ) sizeof ( float ); // d_size = data size.
    const std::uint32_t b_size = ( std::uint32_t ) sizeof ( fmat ) + d_size;  // b_size = total size in bytes of object.

    fmat *m = ( fmat * ) scalable_aligned_malloc ( b_size, _ALIGN );

    if ( m == nullptr ) {

        return nullptr;
    }

    m->n_rows = n_rows;
    m->n_cols = n_cols;
    m->n_elem = n_elem;
    m->d_size = d_size;
    m->b_size = b_size;

    m->is_square = n_rows == n_cols;

    m->v = ( float * ) ( ( ( char * ) m ) + sizeof ( fmat ) ); // Past the parameters...

    LAF_CHECK_DATA_ALIGNMENT ( &m->v [ 0 ], _ALIGN );

    return m;
}


fmat *linear_algebra_float_malloc_memptr ( float *memptr, const std::uint32_t n_rows, const std::uint32_t n_cols ) {

    const std::uint32_t n_elem = n_rows * n_cols;

    if ( n_elem == 0 ) {

        return nullptr;
    }

    fmat *m = ( fmat * ) scalable_aligned_malloc ( sizeof ( fmat ), _ALIGN );

    if ( m == nullptr ) {

        return nullptr;
    }

    m->n_rows = n_rows;
    m->n_cols = n_cols;
    m->n_elem = n_elem;
    m->d_size = n_elem * ( std::uint32_t ) sizeof ( float ); // d_size = data size.
    m->b_size = ( std::uint32_t ) sizeof ( fmat );

    m->is_square = n_rows == n_cols;

    m->v = memptr;

    return m;
}


float linear_algebra_float_min ( const fmat *m ) {

    if ( m == nullptr ) {

        return NAN;
    }

    switch ( m->n_elem ) {

        case 0: return NAN;
        case 1:	return m->v [ 0 ];
        case 2:	return fminf ( m->v [ 0 ], m->v [ 1 ] );
        case 3:

        {
            const float *v = m->v;

            return fminf ( fminf ( v [ 0 ], v [ 1 ] ), v [ 2 ] );
        }

        default:

        {
            float min = FLT_MAX;

            const float *v = m->v;
            const float last = m->n_elem;

            for ( std::uint32_t i = 0; i < last; ++i ) {

                if ( v [ i ] < min ) {

                    min = v [ i ];
                }
            }

            return min;
        }
    }
}


float linear_algebra_float_max ( const fmat *m ) {

    if ( m == nullptr ) {

        return NAN;
    }

    switch ( m->n_elem ) {

        case 0: return NAN;
        case 1: return m->v [ 0 ];
        case 2: return fmaxf ( m->v [ 0 ], m->v [ 1 ] );
        case 3:

        {
            const float *v = m->v;

            return fmaxf ( fmaxf ( v [ 0 ], v [ 1 ] ), v [ 2 ] );
        }

        default:

        {
            float max = -FLT_MAX;

            const float *v = m->v;
            const float last = m->n_elem;

            for ( std::uint32_t i = 0; i < last; ++i ) {

                if ( v [ i ] > max ) {

                    max = v [ i ];
                }
            }

            return max;
        }
    }
}


void linear_algebra_float_min_max ( float *min, float *max, const fmat *m ) {

    // Simultaneous min & max using only 3 * N / 2 comparisons...

    if ( m == nullptr || m->n_elem == 0 ) {

        *min = *max = NAN;

        return;
    }

    float tmp_min, tmp_max;

    std::uint32_t n = 1;

    if ( m->n_elem & 1 ) {

        tmp_min = tmp_max = m->v [ 0 ];
    }

    else {

        const float *v = m->v;

        if ( v [ 0 ] > v [ 1 ] ) {

            tmp_max = v [ 0 ]; tmp_min = v [ 1 ];
        }

        else {

            tmp_min = v [ 0 ]; tmp_max = v [ 1 ];
        }

        n = 2;
    }

    const float *v = m->v;
    const float last = m->n_elem;

    for ( std::uint32_t i0 = 0, i1 = 1; i1 < last; ++i0, ++i1 ) {

        if ( v [ i0 ] > v [ i1 ] ) {

            if ( tmp_max < v [ i0 ] ) tmp_max = v [ i0 ];
            if ( tmp_min > v [ i1 ] ) tmp_min = v [ i1 ];
        }

        else {

            if ( tmp_max < v [ i1 ] ) tmp_max = v [ i1 ];
            if ( tmp_min > v [ i0 ] ) tmp_min = v [ i0 ];
        }
    }

    *min = tmp_min;
    *max = tmp_max;
}


int32_t linear_algebra_float_invert ( fmat *m2, const fmat *m1 ) {

    LAF_CHECK_IS_SQUARE ( m1 );
    LAF_CHECK_IS_SQUARE ( m2 );

    //	Calculates the inverse of the n * n matrix X: Y = X^-1.
    //	Does not change the value of X, unless Y = X...

    linear_algebra_float_copy ( m2, m1 );

    int32_t *lapack_ipiv = ( int32_t * ) scalable_aligned_malloc ( m2->n_rows * sizeof ( int32_t ), _ALIGN );

    int32_t info = LAPACKE_sgetrf ( CblasColMajor, m2->n_rows, m2->n_cols, m2->v, m2->n_rows, lapack_ipiv );

    if ( not ( info ) ) {

        info = LAPACKE_sgetri ( CblasColMajor, m2->n_rows, m2->v, m2->n_rows, lapack_ipiv );
    }

    scalable_aligned_free ( lapack_ipiv );

    return info;
}


int32_t linear_algebra_float_solve ( fmat *vx, const fmat *ma, const fmat *vb ) {

    LAF_CHECK_IS_SQUARE ( ma );

    // Solves the matrix equation A*x = B for x. A is an n * n matrix in column major vector notation.
    // B and x are vectors of lenght n. A and B are not changed. x will contain the solution.
    // x can equal B if B may be overwritten...

    linear_algebra_float_copy ( vx, vb );

    int32_t *lapack_ipiv = ( int32_t * ) scalable_aligned_malloc ( ma->n_rows * sizeof ( int32_t ), _ALIGN );

    const int32_t info = LAPACKE_sgesv ( CblasColMajor, ma->n_rows, 1, ma->v, ma->n_rows, lapack_ipiv, vx->v, ma->n_rows );

    scalable_aligned_free ( lapack_ipiv );

    return info;
}


typedef sax::SRWLock<false> Lock;
typedef std::lock_guard<Lock> ScopedLock;


Lock global_lock;


inline uint64_t linear_algebra_float_random_device ( ) {
    ScopedLock lock_the ( global_lock );
    return sax::os_seed ( );
}


thread_local sax::Rng rng;

#if LAF_MAKE_CHECKS
thread_local bool rng_seeded; // zero initialized, i.e. false...
#endif


void linear_algebra_float_seed ( const std::uint32_t seed_ ) {
#if LAF_MAKE_CHECKS
    rng_seeded = true;
#endif
    rng.seed ( seed_ );
}


void linear_algebra_float_seed ( ) {

#if LAF_MAKE_CHECKS

    if ( not ( rng_seeded ) ) {

        linear_algebra_float_seed ( linear_algebra_float_random_device ( ) );
    }

#else

    linear_algebra_float_seed ( linear_algebra_float_random_device ( ) );

#endif

}


float linear_algebra_float_uniform ( const float lb, const float ub ) {

#if LAF_MAKE_CHECKS

    if ( not ( rng_seeded ) ) {

        linear_algebra_float_seed ( linear_algebra_float_random_device ( ) );
    }

#endif

    return boost::random::uniform_real_distribution<float> ( lb, ub ) ( rng );
}


void linear_algebra_float_uniform ( fmat *m, const float lb, const float ub ) {

#if LAF_MAKE_CHECKS

    if ( not ( rng_seeded ) ) {

        linear_algebra_float_seed ( linear_algebra_float_random_device ( ) );
    }

#endif

    boost::random::uniform_real_distribution<float> dist ( lb, ub );

    const std::uint32_t last = m->n_elem;
    float *v = m->v;

    for ( size_t i = 0; i < last; ++i ) {

        while ( abs ( v [ i ] = dist ( rng ) ) < 0.1f );
    }
}

void linear_algebra_float_gaussian ( fmat *m, const float me, const float sd ) {

#if LAF_MAKE_CHECKS

    if ( not ( rng_seeded ) ) {

        linear_algebra_float_seed ( linear_algebra_float_random_device ( ) );
    }

#endif

    boost::random::normal_distribution<float> dist ( me, sd );

    const std::uint32_t last = m->n_elem;
    float *v = m->v;

    for ( size_t i = 0; i < last; ++i ) {

        while ( abs ( v [ i ] = dist ( rng ) ) < 0.1f );
    }
}


inline std::uint32_t float_as_uint32_t ( float && v_ ) noexcept {

    std::uint32_t r;

    std::memcpy ( &r, &v_, sizeof ( float ) );

    return r;
}

std::uint32_t linear_algebra_float_hash ( const fmat *m ) {

    const std::uint32_t last = m->n_elem;
    const std::uint32_t *v = ( std::uint32_t * ) m->v;

    std::uint32_t h = sax::hash ( m->n_rows ) ^ sax::hash ( sax::hash ( m->n_cols ) * 0x45D9F3B );

    for ( size_t i = 0; i < last; ++i ) {

        h ^= sax::hash ( float_as_uint32_t ( v [ i ] ) );
    }

    return h;
}
