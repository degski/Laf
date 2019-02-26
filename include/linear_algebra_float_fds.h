
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

// Linear Algebra Floats Data Sets...

#include "linear_algebra_float.h"


extern "C" {

	typedef enum { iris, parity_4, yeast, estoxx50 } fds_selector;

	void *fds_get ( const fds_selector s, std::uint32_t *i, std::uint32_t *o, std::uint32_t *tr_p, float **tr_i, float **tr_o, std::uint32_t *va_p, float **va_i, float **va_o, std::uint32_t *te_p, float **te_i, float **te_o );
	void fds_free ( void *p );

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


typedef struct sds {

	std::uint32_t n_patt = 0;

	fmat *iput = nullptr, *oput = nullptr;

} sds;


typedef struct fds {

	void *fds_cppptr = nullptr;

	std::uint32_t n_iput, n_oput;

	sds training, validation, test;

} fds;


fds *linear_algebra_float_data_set_malloc ( const fds_selector s );
