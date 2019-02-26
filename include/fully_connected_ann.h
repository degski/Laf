
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

#include <cstdlib>
#include <string>

#include "linear_algebra_float.h"
#include "linear_algebra_float_fds.h"
#include "activation.h"


// Fully Connected Cascade Network...


#define fully_connected_cascade_free( nn ) { \
\
	for ( std::uint32_t i = 0; i < ( nn )->n_nrns; ++i ) \
\
		linear_algebra_float_free ( ( nn )->lyr [ i ] ); \
\
	linear_algebra_float_free ( ( nn )->ibo ); \
	linear_algebra_float_free ( ( nn )->wts ); \
\
	scalable_aligned_free ( ( nn ) ); \
\
	( nn ) = nullptr; \
}

#define fully_connected_cascade_levenberg_marquardt_free( lm ) { \
\
	linear_algebra_float_free ( ( lm )->ihs ); \
	linear_algebra_float_free ( ( lm )->hes ); \
	linear_algebra_float_free ( ( lm )->gra ); \
	linear_algebra_float_free ( ( lm )->acc ); \
	linear_algebra_float_free ( ( lm )->jpm ); \
	linear_algebra_float_free ( ( lm )->nbn ); \
\
	scalable_aligned_free ( ( lm ) ); \
\
	( lm ) = nullptr; \
}


typedef struct fcn {

	std::uint32_t n_nrns, n_iput, n_oput, n_wgts, ibo_o0, nbn_o0;

	float alpha;
	bool is_recurrent_first_feed_forward = true;

	float ( *activation_function            ) ( const float, const float );
	float ( *activation_function_derivative ) ( const float, const float );

	fmat *wts = nullptr, *ibo = nullptr;

	fmat* lyr [ 0 ];

} fcn;


typedef struct flm {

	float lambda, v, lambda_min, lambda_max, bay_beta, bay_alpha;

	fmat *nbn = nullptr, *jpm = nullptr, *acc = nullptr, *gra = nullptr, *hes = nullptr, *ihs = nullptr;

	float f [ 0 ];

} flm;


fcn *fully_connected_cascade_malloc ( std::uint32_t n_nrns, const std::uint32_t n_iput, const std::uint32_t n_oput, const activation_function activation_function, const float alpha );
fcn *fully_connected_recurrent_cascade_malloc ( std::uint32_t n_nrns, const std::uint32_t n_iput, const std::uint32_t n_oput, const activation_function activation_function, const float alpha );

fcn *fully_connected_cascade_copy ( fcn *nn );

inline void fully_connected_cascade_io_malloc ( fcn *nn, const sds *ss ) {

	nn->ibo = linear_algebra_float_malloc ( ss->n_patt, nn->n_iput + 1 + nn->n_nrns );

	linear_algebra_float_zeros ( nn->ibo );
	linear_algebra_float_copy ( nn->ibo, ss->iput );
}

#define fully_connected_cascade_io_free( nn ) { \
\
	linear_algebra_float_free ( ( nn )->ibo ); \
}


flm *fully_connected_cascade_levenberg_marquardt_malloc ( const std::uint32_t n_nrns, const std::uint32_t n_iput );

float fully_connected_cascade_sse ( const sds *dt, const fmat *ibo );
float fully_connected_cascade_ssw ( const fcn *nn );
float fully_connected_cascade_cle ( const sds *dt, const fmat *ibo );

void fully_connected_cascade_accumulator_fill ( const fmat *v, const std::uint32_t n_elem, fmat *accumulator );
void fully_connected_cascade_accumulator_unfold ( const fmat *accumulator, fmat *m );


namespace {

	inline void fully_connected_cascade_feedforward ( fcn *nn, float *ibo_v, const std::uint32_t ibo_n_rows, const std::uint32_t n_nrns ) {

		for ( std::uint32_t i = 0, n_cols = nn->n_iput + 1; i < n_nrns; ++i, ++n_cols ) {

			cblas_sgemv ( CblasColMajor, CblasNoTrans, ibo_n_rows, n_cols, 1.0f, ibo_v, ibo_n_rows, nn->lyr [ i ]->v, 1, 0.0f, ibo_v + n_cols * ibo_n_rows, 1 );

			linear_algebra_float_apply_binary_function_bind2nd ( ibo_v + n_cols * ibo_n_rows, ibo_n_rows, nn->activation_function, nn->alpha );
		}
	}
}

inline void fully_connected_cascade_feedforward ( fcn *nn ) {

	fully_connected_cascade_feedforward ( nn, nn->ibo->v, nn->ibo->n_rows, nn->n_nrns );
}

inline void fully_connected_recurrent_cascade_feedforward ( fcn *nn ) {

	float *ibo_v = nn->ibo->v;
	const std::uint32_t ibo_n_rows = nn->ibo->n_rows, n_nrns = nn->n_nrns;
	float *ibo_v_n_iput_1_ibo_n_rows = ibo_v + ( nn->n_iput + 1 ) * ibo_n_rows;

	if ( nn->is_recurrent_first_feed_forward ) {

		nn->is_recurrent_first_feed_forward = false;

		memset ( ibo_v_n_iput_1_ibo_n_rows - n_nrns * ibo_n_rows, 0, n_nrns * sizeof ( float ) * ibo_n_rows );
	}

	else {

		memcpy ( ibo_v_n_iput_1_ibo_n_rows - n_nrns * ibo_n_rows, ibo_v_n_iput_1_ibo_n_rows, n_nrns * sizeof ( float ) * ibo_n_rows );
	}

	fully_connected_cascade_feedforward ( nn, ibo_v, ibo_n_rows, n_nrns );
}


void fully_connected_cascade_hessian_gradients ( const fcn *nn, flm *lm, const sds *ds );

float fully_connected_cascade_levenberg_marquardt_epoch ( fcn *nn, flm *lm, const sds *ds );
float fully_connected_cascade_levenberg_marquardt_baysian_epoch ( fcn *nn, flm *lm, const sds *ds );

float fully_connected_cascade_train ( fcn *nn, const fds *ds );
float fully_connected_cascade_baysian_train ( fcn *nn, const fds *ds );

void fully_connected_cascade_print_errs ( fcn *nn, flm *lm, const sds *ss, const char *s );
void fully_connected_cascade_print_wgts ( const fcn *nn );
