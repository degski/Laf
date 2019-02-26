
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

// Simplified Fully Connected...

#include <cstdlib>
#include <string>

#include "linear_algebra_float.h"
#include "linear_algebra_float_fds.h"
#include "activation.h"



//	Simplified implementation of the fcc neural network architecture (Simplified Fully Connected)...
//	Only one input sample...


#define simplified_fully_connected_cascade_nn_free( nn ) { \
\
	for ( std::uint32_t i = 0; i < ( nn )->n_nrns; i++ ) { \
\
		linear_algebra_float_free ( ( nn )->lyr [ i ].g ); \
		linear_algebra_float_free ( ( nn )->lyr [ i ].w ); \
\
	} \
\
	linear_algebra_float_free ( ( nn )->grd ); \
	linear_algebra_float_free ( ( nn )->wts ); \
\
	linear_algebra_float_free ( ( nn )->dlt ); \
	linear_algebra_float_free ( ( nn )->sop ); \
	linear_algebra_float_free ( ( nn )->ibo ); \
\
	scalable_aligned_free ( ( nn ) ); \
\
	( nn ) = nullptr; \
}


typedef struct wgp {

	fmat* w = nullptr;
	fmat* g = nullptr;

} wgp;


typedef struct sfc {

	std::uint32_t n_nrns, n_iput, n_oput, n_wgts, ibo_o0, nbn_o0;

	float alpha;

	float ( *activation_function            ) ( const float, const float );
	float ( *activation_function_derivative ) ( const float, const float );

	fmat *ibo = nullptr, *sop = nullptr, *dlt = nullptr, *wts = nullptr, *grd = nullptr;

	wgp lyr [ 0 ];

} sfc;


sfc *simplified_fully_connected_cascade_nn_malloc ( std::uint32_t n_nrns, const std::uint32_t n_iput, const std::uint32_t n_oput, const activation_function activation_function, const float alpha );


inline void simplified_fully_connected_cascade_get_sample ( sfc *nn, const sds *ss, const std::uint32_t p ) {

	linear_algebra_float_zeros ( nn->ibo );
	linear_algebra_float_copy_row_vec ( nn->ibo, ss->iput, p );
	linear_algebra_float_copy_row_vec ( nn->sop, ss->oput, p );
}


inline void simplified_fully_connected_cascade_feedforward ( sfc *nn ) {

	float *ibo_v = nn->ibo->v;
	const float alpha = nn->alpha;
	const std::uint32_t n_nrns = nn->n_nrns;

	for ( std::uint32_t i = 0, col = nn->n_iput + 1; i < n_nrns; i++, col++ ) {

		ibo_v [ col ] = nn->activation_function ( cblas_sdot ( col, ibo_v, 1, nn->lyr [ i ].w->v, 1 ), alpha );
	}
}


inline void simplified_fully_connected_cascade_backpropagate ( sfc *nn ) {

	float *ibo_v = nn->ibo->v, *sop_v = nn->sop->v, dlt, *wts_v = nullptr;
	const float learning = 0.9f, alpha = nn->alpha;
	const std::uint32_t n_iput = nn->n_iput +1;

	for ( std::uint32_t col = nn->ibo->n_cols -1; col >= n_iput; col-- ) {

		dlt = ( sop_v [ col - n_iput ] - ibo_v [ col ] ) * nn->activation_function_derivative ( ibo_v [ col ], alpha );

		wts_v = nn->lyr [ col - n_iput ].w->v;

		for ( std::uint32_t i = 0; i < col; i++ ) {

			wts_v [ i ] += learning * dlt * ibo_v [ i ];
		}

		col--;

		// olay.dlt = ( aset.oput - olay.oput ) % derivative_unipolar ( olay.oput );
		// hlay.dlt = ( ( aset.oput - olay.oput ) % derivative_unipolar ( olay.oput ) * olay.wts.rows ( 0, olay.wts.n_rows -2 ).t ( ) ) % derivative_unipolar ( hlay.oput );
		dlt = ( sop_v [ col - n_iput ] - ibo_v [ col ] ) * nn->activation_function_derivative ( ibo_v [ col ], alpha );

		wts_v = nn->lyr [ col - n_iput ].w->v;
	}
}

#if 0

	void backpropagate ( const aset &aset ) { // 4, 7, 3

		// vanilla

		// output layer
		olay.dlt = ( aset.oput - olay.oput ) % derivative_unipolar ( olay.oput ); // error signal: 75x3
		olay.grd.slice ( 0 ) = hlay.oput.t ( ) * olay.dlt; // 8x75 * 75x3 = 8x3
		olay.wts += learning * olay.grd.slice ( 0 );

		// hidden layer
		hlay.dlt = ( olay.dlt * olay.wts.rows ( 0, olay.wts.n_rows -2 ).t ( ) ) % derivative_unipolar ( hlay.oput ); // 75x3 * 3x7 = 75x7 % 75x8 = 75x7
		hlay.grd.slice ( 0 ) = aset.iput.t ( ) * hlay.dlt; // 5x75 * 75x7 = 5x7
		hlay.wts += learning * hlay.grd.slice ( 0 );
	}

#endif

void simplified_fully_connected_cascade_print_wgts ( const sfc *nn );
