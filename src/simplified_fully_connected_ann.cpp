
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

#include "simplified_fully_connected_ann.h"
#include <float.h>
#include <stdbool.h>

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

inline void simplified_fully_connected_cascade_nguyen_widrow_weights ( fmat *w, const std::uint32_t n_iput, const std::uint32_t n_oput ) {

	// Nguyen-Widrow weight initialization

	const float beta = 0.7f * powf ( ( float ) n_oput, 1.0f / ( float ) ( n_iput +1 ) );

	linear_algebra_float_uniform ( w, -1.0f, 1.0f );

	for ( std::uint32_t c = 0; c < w->n_elem; c++ )

		w->v [ c ] *= beta / linear_algebra_float_norm ( w );
}


sfc *simplified_fully_connected_cascade_nn_malloc ( std::uint32_t n_nrns, const std::uint32_t n_iput, const std::uint32_t n_oput, const activation_function activation_function, const float alpha ) {

	n_nrns = max ( n_nrns, n_oput );

	sfc *r = ( sfc * ) scalable_aligned_malloc ( sizeof ( sfc ) + n_nrns * sizeof ( wgp ), _ALIGN );

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

	r->ibo = linear_algebra_float_malloc ( 1, n_iput_1 + n_nrns );
	r->sop = linear_algebra_float_malloc ( 1, n_oput );
	r->dlt = linear_algebra_float_malloc ( 1, n_iput_1 + n_nrns );

	r->wts = linear_algebra_float_malloc ( n_wgts, 1 );
	r->grd = linear_algebra_float_malloc ( n_wgts, 1 );

	#if 1
	simplified_fully_connected_cascade_nguyen_widrow_weights ( r->wts, r->n_iput, r->n_oput );
	#else
	linear_algebra_float_uniform ( r->wts, -1.0f, 1.0f );
	#endif

	float *r_wts_v = r->wts->v, *r_grd_v = r->grd->v;

	for ( std::uint32_t i = 0; i < ( const std::uint32_t ) n_nrns; i++ ) {

		const std::uint32_t n_iput_1_i = n_iput_1 + i;

		r->lyr [ i ].w = linear_algebra_float_malloc_memptr ( r_wts_v, n_iput_1_i, 1 );
		r->lyr [ i ].g = linear_algebra_float_malloc_memptr ( r_grd_v, n_iput_1_i, 1 );

		r_wts_v += n_iput_1_i;
		r_grd_v += n_iput_1_i;
	}

	return r;
}


void simplified_fully_connected_cascade_print_wgts ( const sfc *nn ) {

	fmat *w = linear_algebra_float_calloc ( nn->n_nrns + nn->n_iput, nn->n_nrns );

	for ( std::uint32_t i = 0; i < ( const std::uint32_t ) nn->n_nrns; i++ ) {

		memcpy ( w->v + i * w->n_rows, nn->lyr [ i ].w->v, nn->lyr [ i ].w->n_elem * sizeof ( float ) );
	}

	linear_algebra_float_print ( w, 3 );

	linear_algebra_float_free ( w );
}
