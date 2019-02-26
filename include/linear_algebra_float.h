
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

// Linear Algebra Floats...

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif


#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_trans.h>

#include <iostream>

// #include <data_sets.h>

#ifndef LAF_MAKE_CHECKS
#define LAF_MAKE_CHECKS 1
#endif

#ifndef OMP_NUM_THREADS
#define OMP_NUM_THREADS getenv ( "OMP_NUM_THREADS" )
#endif
#define OMP_LOAD 4


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


#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))


#define _ALIGN 64
#define MEMORY_ALIGN __declspec ( align ( _ALIGN ) )


#define LAF_SWAP_DIMENSIONS( m ) { const std::uint32_t t = ( m )->n_cols; ( m )->n_cols = ( m )->n_rows; ( m )->n_rows = t; }


#if LAF_MAKE_CHECKS

#define LAF_CHECK_IS_SQUARE( m ) { \
\
	if ( ( m )->is_square == false || ( m )->n_elem == 0 ) { \
\
		printf ( "matrix not square (or zero).\n" ); \
\
		exit ( 0 ); \
	} \
}

#define LAF_CHECK_VEC_SIZE( m, v ) { \
\
	if ( ( m )->n_rows > ( v )->n_elem ) { \
\
		printf ( "vector not large enough.\n" ); \
\
		exit ( 0 ); \
	} \
}

#define LAF_CHECK_ROWVEC_SIZE( m, v ) { \
\
	if ( ( m )->n_cols > ( v )->n_elem ) { \
\
		printf ( "vector not large enough.\n" ); \
\
		exit ( 0 ); \
	} \
}

#define LAF_CHECK_MAT_SIZE_EQUALITY( m1, m2 ) { \
\
	if ( ( m1 )->n_elem != ( m2 )->n_elem ) { \
\
		printf ( "matrices not same size.\n" ); \
\
		linear_algebra_float_print ( m1, 3 ); \
		linear_algebra_float_print ( m2, 3 ); \
\
		exit ( 0 ); \
	} \
}

#define LAF_CHECK_SIZE( m, size ) { \
\
	if ( ( m )->n_elem != ( size ) ) { \
\
		printf ( "matrix not correct size.\n" ); \
\
		exit ( 0 ); \
	} \
}

std::uint32_t linear_algebra_float_pointer_alignment ( const void *p );

#define LAF_CHECK_DATA_ALIGNMENT( memptr, align ) { \
\
	if ( linear_algebra_float_pointer_alignment ( memptr ) < align ) { \
\
		printf ( "data pointer not correctly aligned (%i).\n", ( int ) linear_algebra_float_pointer_alignment ( memptr ) ); \
\
		exit ( 0 ); \
	} \
}

#define LAF_ZEROS( m ) linear_algebra_float_zeros ( ( m ) )

#else

#define LAF_CHECK_IS_SQUARE( m )
#define LAF_CHECK_VEC_SIZE( m, v )
#define LAF_CHECK_ROWVEC_SIZE( m, v )
#define LAF_CHECK_MAT_SIZE_EQUALITY( m1, m2 )
#define LAF_CHECK_SIZE( m, size )
#define LAF_CHECK_DATA_ALIGNMENT( memptr, align )
#define LAF_ZEROS( m )

#endif // LAF_MAKE_CHECKS


// free

#define linear_algebra_float_free( m ) { \
\
	if ( ( m ) != nullptr ) { \
\
	    scalable_aligned_free ( ( m ) ); \
\
	    ( m ) = nullptr; \
	} \
}


#define linear_algebra_float_data_set_free( ds ) { \
\
	linear_algebra_float_free (       ( ds )->test.oput ); \
	linear_algebra_float_free (       ( ds )->test.iput ); \
	linear_algebra_float_free ( ( ds )->validation.oput ); \
	linear_algebra_float_free ( ( ds )->validation.iput ); \
	linear_algebra_float_free (   ( ds )->training.oput ); \
	linear_algebra_float_free (   ( ds )->training.iput ); \
\
	fds_free ( ( ds )->fds_cppptr ); \
\
	( ds )->fds_cppptr = nullptr; \
\
	scalable_aligned_free ( ( ds ) ); \
\
	( ds ) = nullptr; \
}


#define linear_algebra_float_runtime_array_free( m ) { \
\
	if ( ( m ) != nullptr ) { \
\
	    scalable_aligned_free ( ( m ) ); \
\
	    ( m ) = nullptr; \
	} \
}


typedef struct fmat { // 32 bytes...

	std::uint32_t n_rows = 0, n_cols = 0, n_elem = 0, d_size = 0, b_size = 0; // 20 bytes...

	bool is_square = false; // 4 bytes...

	float *v = nullptr; // 8 bytes...

	float &at ( const std::uint32_t i, const std::uint32_t j ) {

		return v [ i * n_cols + j ];
	}

	float at ( const std::uint32_t i, const std::uint32_t j ) const {

		return v [ i * n_cols + j ];
	}

} fmat;


inline std::uint32_t linear_algebra_float_padded ( const std::uint32_t v, const std::uint32_t alignment ) {

	return ( v + alignment - 1 ) & ~( alignment - 1 );
}


inline std::uint32_t linear_algebra_float_padding ( const std::uint32_t v, const std::uint32_t alignment ) {

	return linear_algebra_float_padded ( v, alignment ) - v;
}


inline std::uint32_t linear_algebra_float_shift ( const std::uint32_t v, const std::uint32_t alignment ) {

	// Calculate how much the ptr/data-array needs to be shifted to align the data array part...

	return alignment - linear_algebra_float_padding ( v, alignment );
}


inline std::uint32_t linear_algebra_float_pointer_alignment ( const void *p ) {

	return ( std::uint32_t ) ( ( std::uintptr_t ) p & ( std::uintptr_t ) -( ( std::intptr_t ) p ) );
}


inline bool linear_algebra_float_verify_alignment ( const void *p, const std::uintptr_t n ) {

	return ( ( std::uintptr_t ) p & ( n - 1 ) ) != 0;
}


fmat *linear_algebra_float_malloc ( const std::uint32_t n_rows, const std::uint32_t n_cols );

inline void linear_algebra_float_zeros ( fmat *m );


inline fmat *linear_algebra_float_calloc ( const std::uint32_t n_rows, const std::uint32_t n_cols ) {

	fmat *m = linear_algebra_float_malloc ( n_rows, n_cols );

	linear_algebra_float_zeros ( m );

	return m;
}


fmat *linear_algebra_float_malloc_memptr ( float *memptr, const std::uint32_t n_rows, const std::uint32_t n_cols );


float linear_algebra_float_min ( const fmat *m );
float linear_algebra_float_max ( const fmat *m );
void linear_algebra_float_min_max ( float *min, float *max, const fmat *m );


void linear_algebra_float_print ( const fmat *m, const std::uint32_t precision );
void linear_algebra_float_print_transposed ( fmat *m, const std::uint32_t precision );
void linear_algebra_float_print_rm ( const fmat *m, const std::uint32_t precision );


int32_t linear_algebra_float_invert ( fmat *m2, const fmat *m1 );
int32_t linear_algebra_float_solve ( fmat *vx, const fmat *ma, const fmat *vb );


void linear_algebra_float_seed ( const std::uint32_t seed );
void linear_algebra_float_seed ( );


float linear_algebra_float_uniform ( const float lb, const float ub );
void linear_algebra_float_uniform ( fmat *m, const float lb, const float ub );
void linear_algebra_float_gaussian ( fmat *m, const float me, const float sd );


inline void linear_algebra_float_zeros ( fmat *m ) {

	std::memset ( m->v, 0, m->d_size );
}


inline void linear_algebra_float_ones ( fmat *m ) {

	const std::uint32_t n_elem = m->n_elem;
	float *v = m->v;

	for ( std::uint32_t i = 0; i < n_elem; ++i ) {

		v [ i ] = 1.0f;
	}
}


inline void linear_algebra_float_fill ( fmat *m, const float f ) {

	const std::uint32_t n_elem = m->n_elem;
	float *v = m->v;

	for ( std::uint32_t i = 0; i < n_elem; ++i ) {

		v [ i ] = f;
	}
}


std::uint32_t linear_algebra_float_hash ( const fmat *m );


inline void linear_algebra_float_copy ( fmat *m2, const fmat *m1 ) {

	#if defined ( _DEBUG )

	if ( m2->n_elem == 0 ) {

		std::cout << "laf copy: number of elements 0.\n";
	}

	if ( m2->v == m1->v ) {

		std::cout << "laf copy: source and destination the same.\n";
	}

	if ( m2->v == nullptr ) {

		std::cout << "laf copy: destination not allocated.\n";
	}

	if ( m1->v == nullptr ) {

		std::cout << "laf copy: source not allocated.\n";
	}

	#endif

	memcpy ( m2->v, m1->v, MIN ( m1->d_size, m2->d_size ) );
}


inline void linear_algebra_float_copy_col ( fmat *m, const std::uint32_t col, const fmat *v ) {

	LAF_CHECK_SIZE ( m, v->n_rows );

	memcpy ( m->v + col * m->n_rows, v->v, m->d_size );
}


inline void linear_algebra_float_copy_row_vec ( fmat *v, const fmat *m, const std::uint32_t row ) {

	LAF_CHECK_ROWVEC_SIZE ( m, v );

	const float *v_last = v->v + m->n_cols;
	const std::uint32_t m_n_rows = m->n_rows;

	for ( float *v_v = v->v, *m_v_r = m->v + row; v_v < v_last; ++v_v, m_v_r += m_n_rows ) {

		*v_v = *m_v_r;
	}
}


inline void linear_algebra_float_copy_cols ( fmat *m2, const fmat *m1, const std::uint32_t left_col_idx, const std::uint32_t right_col_idx ) {

	LAF_CHECK_SIZE ( m2, ( ( right_col_idx - left_col_idx + 1 ) * m1->n_rows ) );

	memcpy ( m2->v, m1->v + left_col_idx * m1->n_rows, ( ( right_col_idx - left_col_idx + 1 ) * m1->n_rows ) * sizeof ( float ) );
}


inline void linear_algebra_float_copy_dia ( fmat *v, const fmat *m ) {

	// Copy diagonal of matrix to a vector...

	LAF_CHECK_IS_SQUARE ( m );
	LAF_CHECK_VEC_SIZE ( m, v );

	std::uint32_t i = 0;
	const size_t last = v->n_elem;
	float *v_v = v->v;
	const float *m_v = m->v;
	const std::uint32_t n_rows_1 = m->n_rows + 1;

	for ( ; i < last; ++i ) {

		v_v [ i ] = m_v [ i * n_rows_1 ];
	}
}


inline void linear_algebra_float_dia_copy_add ( fmat *m, const fmat *v, const float f ) {

	// Copy a vector to the diagonal of matrix and element-wise add a constant...

	LAF_CHECK_IS_SQUARE ( m );
	LAF_CHECK_VEC_SIZE ( m, v );

	std::uint32_t i = 0;
	const size_t last = v->n_elem;
	float *v_v = v->v;
	float *m_v = m->v;
	const std::uint32_t n_rows_1 = m->n_rows + 1;

	for ( ; i < last; ++i ) {

		m_v [ i * n_rows_1 ] = v_v [ i ] + f;
	}
}


inline float linear_algebra_float_trace ( const fmat *m ) {

	LAF_CHECK_IS_SQUARE ( m );

	float trace = 0.0f;

	std::uint32_t i = 0;
	const size_t last = m->n_rows;
	const float *m_v = m->v;
	const std::uint32_t n_rows_1 = m->n_rows + 1;

	for ( ; i < last; ++i ) {

		trace += m_v [ i * n_rows_1 ];
	}

	return trace;
}


inline void linear_algebra_float_in_place_transpose ( fmat *m ) {

	mkl_simatcopy ( 'C', 'T', m->n_rows, m->n_cols, 1.0f, m->v, m->n_rows, m->n_cols );

	LAF_SWAP_DIMENSIONS ( m );
}


inline void linear_algebra_float_out_place_transpose ( const fmat *m1, fmat *m2 ) {

	LAF_CHECK_MAT_SIZE_EQUALITY ( m1, m2 );

	mkl_somatcopy ( 'C', 'T', m1->n_rows, m1->n_cols, 1.0f, m1->v, m1->n_rows, m2->v, m2->n_rows );
}


inline void linear_algebra_float_transpose ( const fmat *m1, fmat *m2 ) {

	if ( m1 == m2 ) {

		linear_algebra_float_in_place_transpose ( m2 );
	}

	else {

		linear_algebra_float_out_place_transpose ( m1, m2 );
	}
}


inline void linear_algebra_float_add ( fmat *m, const fmat *a ) {

	LAF_CHECK_MAT_SIZE_EQUALITY ( m, a );

	const float last = m->n_elem;

	float *m_v = m->v;
	const float *a_v = a->v;

	for ( std::uint32_t i = 0; i < last; ++i ) {

		m_v [ i ] += a_v [ i ];
	}
}


inline void linear_algebra_float_sub ( fmat *m, const fmat *a ) {

	LAF_CHECK_MAT_SIZE_EQUALITY ( m, a );

	const float last = m->n_elem;

	float *m_v = m->v;
	const float *a_v = a->v;

	for ( std::uint32_t i = 0; i < last; ++i ) {

		m_v [ i ] -= a_v [ i ];
	}
}


inline void linear_algebra_float_sub_with_decay ( fmat *m, const fmat *a, const float decay ) {

	// Subtract one matrix from another element-wise and post multiply by a scalar...

	LAF_CHECK_MAT_SIZE_EQUALITY ( m, a );

	const float last = m->n_elem;

	float *m_v = m->v;
	const float *a_v = a->v;

	for ( std::uint32_t i = 0; i < last; ++i ) {

		m_v [ i ] -= a_v [ i ];
		m_v [ i ] *= decay;
	}
}


inline void linear_algebra_float_mul_mv ( fmat *r, const fmat *m, const fmat *v ) {

	cblas_sgemv ( CblasColMajor, CblasNoTrans, m->n_rows, m->n_cols, 1.0f, m->v, m->n_rows, v->v, 1, 0.0f, r->v, 1 );
}


inline void linear_algebra_float_mul_rm ( fmat *r, const fmat *v, const fmat *m ) {

	cblas_sgemv ( CblasColMajor, CblasTrans, m->n_rows, m->n_cols, 1.0f, m->v, m->n_rows, v->v, 1, 0.0f, r->v, 1 );
}


inline void linear_algebra_float_mul_mm ( fmat *r, const fmat *m1, const fmat *m2 ) {

	cblas_sgemm ( CblasColMajor, CblasNoTrans, CblasNoTrans, m1->n_rows, m2->n_cols, m1->n_cols, 1.0f, m1->v, m1->n_rows, m2->v, m2->n_rows, 0.0f, r->v, r->n_rows );
}


inline float linear_algebra_float_dot ( const fmat *m1, const fmat *m2 ) {

	LAF_CHECK_MAT_SIZE_EQUALITY ( m1, m2 );

	return cblas_sdot ( m1->n_elem, m1->v, 1, m2->v, 1 );
}


inline float linear_algebra_float_norm ( const fmat *m ) {

	return sqrtf ( cblas_sdot ( m->n_elem, m->v, 1, m->v, 1 ) );
}


inline float linear_algebra_float_norm_sqr ( const fmat *m ) {

    return cblas_sdot ( m->n_elem, m->v, 1, m->v, 1 );
}


inline void linear_algebra_float_apply_unary_function ( float *v, const std::uint32_t n_elem, float ( *unary_function ) ( const float ) ) {

	for ( std::uint32_t i = 0; i < n_elem; ++i ) {

		v [ i ] = unary_function ( v [ i ] );
	}
}


inline void linear_algebra_float_apply_binary_function_bind1st ( float *v, const std::uint32_t n_elem, float ( *binary_function ) ( const float, const float ), const float first ) {

	// first parameter is bound on call

	for ( std::uint32_t i = 0; i < n_elem; ++i ) {

		v [ i ] = binary_function ( first, v [ i ] );
	}
}


inline void linear_algebra_float_apply_binary_function_bind2nd ( float *v, const std::uint32_t n_elem, float ( *binary_function ) ( const float, const float ), const float second ) {

	// second parameter is bound on call

	for ( std::uint32_t i = 0; i < n_elem; ++i ) {

		v [ i ] = binary_function ( v [ i ], second );
	}
}


inline void linear_algebra_float_mul ( fmat *m, const float a ) {

	const std::uint32_t last = m->n_elem;

	float *v = m->v;

	for ( std::uint32_t i = 0; i < last; ++i ) {

		v [ i ] *= a;
	}
}


// Offset for m [ i ][ j ], w is an row width, t == 1 for transposed access.

#define mi( w, t, i, j ) ( 4 * ( ( ( i ) * ( w ) + ( j ) ) * ( 1 - ( t ) ) + ( ( j ) * ( w ) + ( i ) ) * ( t ) ) )
