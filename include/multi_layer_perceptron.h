
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

// Multi-Layer Perceptron...

#include <cstdlib>
#include <string>

#include "linear_algebra_float.h"
#include "linear_algebra_float_fds.h"
#include "activation.h"


#define multi_layer_perceptron_nn_free( nn ) { \
\
	for ( std::uint32_t i = 0; i < ( nn )->n_nrns; i++ ) \
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


typedef struct multi_layer_perceptron {

	std::uint32_t n_nrns, n_iput, n_oput, n_wgts, ibo_o0, nbn_o0;

	float alpha;

	float ( *activation_function            ) ( const float, const float );
	float ( *activation_function_derivative ) ( const float, const float );

	fmat *wts = nullptr, *ibo = nullptr;

	fmat* lyr [ 0 ];

} multi_layer_perceptron;


multi_layer_perceptron *multi_layer_perceptron_nn_malloc ( std::uint32_t n_nrns, const std::uint32_t n_iput, const std::uint32_t n_oput, const activation_function activation_function, const float alpha );


inline void multi_layer_perceptron_io_malloc ( multi_layer_perceptron *nn, const sds *ss ) {

	nn->ibo = linear_algebra_float_malloc ( ss->n_patt, nn->n_iput +1 + nn->n_nrns );

	linear_algebra_float_zeros ( nn->ibo );
	linear_algebra_float_copy ( nn->ibo, ss->iput );
}

#define multi_layer_perceptron_io_free( nn ) { \
\
	linear_algebra_float_free ( ( nn )->ibo ); \
}


float multi_layer_perceptron_sse ( const sds *dt, const fmat *ibo );
float multi_layer_perceptron_ssw ( const multi_layer_perceptron *nn );
float multi_layer_perceptron_cle ( const sds *dt, const fmat *ibo );


inline void multi_layer_perceptron_feedforward ( multi_layer_perceptron *nn ) {

	float *ibo_v = nn->ibo->v;

	const std::uint32_t ibo_n_rows = nn->ibo->n_rows, n_nrns = nn->n_nrns;

	for ( std::uint32_t i = 0, n_cols = nn->n_iput +1; i < n_nrns; i++, n_cols++ ) {

		cblas_sgemv ( CblasColMajor, CblasNoTrans, ibo_n_rows, n_cols, 1.0f, ibo_v, ibo_n_rows, nn->lyr [ i ]->v, 1, 0.0f, ibo_v + n_cols * ibo_n_rows, 1 );

		linear_algebra_float_apply_binary_function_bind2nd ( ibo_v + n_cols * ibo_n_rows, ibo_n_rows, nn->activation_function, nn->alpha );
	}
}


void multi_layer_perceptron_print_errs ( multi_layer_perceptron *nn, const sds *ss, const char *s );
void multi_layer_perceptron_print_wgts ( const multi_layer_perceptron *nn );
