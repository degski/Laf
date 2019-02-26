
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

#include "multi_layer_perceptron.h"
#include <float.h>
#include <stdbool.h>


inline void multi_layer_perceptron_nguyen_widrow_weights ( fmat *w, const std::uint32_t n_iput, const std::uint32_t n_oput ) {

	// Nguyen-Widrow weight initialization

	const float beta = 0.7f * powf ( ( float ) n_oput, 1.0f / ( float ) ( n_iput +1 ) );

	linear_algebra_float_uniform ( w, -1.0f, 1.0f );

	for ( std::uint32_t c = 0; c < w->n_elem; c++ ) {

		w->v [ c ] *= beta / linear_algebra_float_norm ( w );
	}
}


multi_layer_perceptron *multi_layer_perceptron_nn_malloc ( std::uint32_t n_nrns, const std::uint32_t n_iput, const std::uint32_t n_oput, const activation_function activation_function, const float alpha ) {

	n_nrns = MAX ( n_nrns, n_oput );

	multi_layer_perceptron *r = ( multi_layer_perceptron * ) scalable_aligned_malloc ( sizeof ( multi_layer_perceptron ) + n_nrns * sizeof ( fmat * ), _ALIGN );

	r->n_nrns = n_nrns;
	r->n_iput = n_iput;
	r->n_oput = n_oput;

	const std::uint32_t n_iput_1 = n_iput +1, n_wgts = ( n_nrns * ( n_nrns + n_iput + n_iput_1 ) ) / 2;

	r->n_wgts = n_wgts;

	r->ibo_o0 = n_iput_1 + n_nrns - n_oput;
	r->nbn_o0 = n_nrns - n_oput;

	r->alpha = alpha;

	switch ( activation_function ) {

		case activation_function::unipolar:

			r->activation_function = activation_unipolar;
			r->activation_function_derivative = activation_unipolar_derivative;

			break;

		case activation_function::bipolar:

			r->activation_function = activation_bipolar;
			r->activation_function_derivative = activation_bipolar_derivative;

			break;

		default:

			printf ( "undefined activation_function\n" );

			exit ( 0 );
	}

	r->ibo = nullptr;
	r->wts = linear_algebra_float_malloc ( n_wgts, 1 );

	#if 1
	multi_layer_perceptron_nguyen_widrow_weights ( r->wts, r->n_iput, r->n_oput );
	#else
	linear_algebra_float_uniform ( r->wts, -1.0f, 1.0f );
	#endif

	float *r_wts_v_m = r->wts->v;

	for ( std::uint32_t i = 0; i < ( const std::uint32_t ) n_nrns; i++ ) {

		const std::uint32_t n_iput_1_i = n_iput_1 + i;

		r->lyr [ i ] = linear_algebra_float_malloc_memptr ( r_wts_v_m, n_iput_1_i, 1 );

		r_wts_v_m += n_iput_1_i;
	}

	return r;
}


float multi_layer_perceptron_sse ( const sds *dt, const fmat *ibo ) {

	// sum-of-squares error

	float sse = 0.0f, *dv = dt->oput->v, *iv = ibo->v;

	const std::uint32_t dr = dt->oput->n_rows, dc = dt->oput->n_cols, id = ibo->n_cols - dt->oput->n_cols;

	const float *dvr = dv + dr;

	for ( std::uint32_t j = 0; j < dc; j++ ) {

		const float *dl = dvr + j * dr;

		for ( float *d = dv + j * dr, *i = iv + ( j + id ) * dr; d < dl; d++, i++ ) {

			const float t = *d - *i;

			sse += t * t;
		}
	}

	return 0.5f * sse / ( float ) dr;
}


float multi_layer_perceptron_ssw ( const multi_layer_perceptron *nn ) {

	// sum-of-squares weight

	const std::uint32_t  n_wgts = nn->n_wgts;
	const float *wts_v = nn->wts->v;

	float ssw = 0.0f;

	for ( std::uint32_t i = 0; i < n_wgts; i++ ) {

		const float w = wts_v [ i ];

		ssw += w * w;
	}

	return 0.5f * ssw / ( float ) n_wgts;
}


float multi_layer_perceptron_cle ( const sds *dt, const fmat *ibo ) {

	// classification error

	std::uint32_t cle = 0;

	const std::uint32_t n_rows = dt->oput->n_rows, n_cols = dt->oput->n_cols;
	float *dv = dt->oput->v, *iv = ibo->v + ( ibo->n_cols - n_cols ) * n_rows;

	for ( std::uint32_t r = 0; r < n_rows; r++ ) {

		float *d = dv++, d_max = *d, *i = iv++, i_max = *i;
		std::uint32_t c = 0, d_idx = 0, i_idx = 0;

		while ( ++c < n_cols ) {

			if ( d += n_rows, *d > d_max ) d_max = *d, d_idx = c;
			if ( i += n_rows, *i > i_max ) i_max = *i, i_idx = c;
		}

		cle += d_idx != i_idx;
	}

	return ( float ) cle / ( float ) n_rows;
}


void multi_layer_perceptron_backpropagate ( const sds * ) { // 4, 7, 3
//	void multi_layer_perceptron_backpropagate ( const sds *ss ) { // 4, 7, 3
	// vanilla

	#if 0

	// output layer
	olay.dlt = ( aset.oput - olay.oput ) % derivative_unipolar ( olay.oput ); // error signal: 75x3
	olay.grd.slice ( 0 ) = hlay.oput.t ( ) * olay.dlt; // 8x75 * 75x3 = 8x3
	olay.wts += learning * olay.grd.slice ( 0 );

	// hidden layer
	hlay.dlt = ( olay.dlt * olay.wts.rows ( 0, olay.wts.n_rows -2 ).t ( ) ) % derivative_unipolar ( hlay.oput ); // 75x3 * 3x7 = 75x7 % 75x8 = 75x7
	hlay.grd.slice ( 0 ) = aset.iput.t ( ) * hlay.dlt; // 5x75 * 75x7 = 5x7
	hlay.wts += learning * hlay.grd.slice ( 0 );

	#endif
}


void multi_layer_perceptron_print_errs ( multi_layer_perceptron *nn, const sds *ss, const char *s ) {

	multi_layer_perceptron_feedforward ( nn );

	const float new_sse = multi_layer_perceptron_sse ( ss, nn->ibo ), new_ssw = multi_layer_perceptron_ssw ( nn );

	printf ( " sse_err %s: %.8f\n", s, new_sse );
	printf ( " ssw_val %s: %.8f\n", s, new_ssw );
	printf ( " cls_err %s: %.8f\n\n", s , multi_layer_perceptron_cle ( ss, nn->ibo ) );
}


void multi_layer_perceptron_print_wgts ( const multi_layer_perceptron *nn ) {

	fmat *w = linear_algebra_float_calloc ( nn->n_nrns + nn->n_iput, nn->n_nrns );

	for ( std::uint32_t i = 0; i < ( const std::uint32_t ) nn->n_nrns; i++ ) {

		memcpy ( w->v + i * w->n_rows, nn->lyr [ i ]->v, nn->lyr [ i ]->n_elem * sizeof ( float ) );
	}

	linear_algebra_float_print ( w, 3 );

	linear_algebra_float_free ( w );
}
