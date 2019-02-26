
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

#include "linear_algebra_float_fds.h"

#include <cfloat> // for FLT_MAX, FLT_MIN
#include <cstdint>

#include <mkl_lapacke.h>


// extern "C" void *
// fds_get ( const fds_selector s, std::uint32_t *i, std::uint32_t *o, std::uint32_t *tr_p, float **tr_i, float **tr_o, std::uint32_t *va_p, float **va_i, float **va_o, std::uint32_t *te_p, float **te_i, float **te_o );


fds *linear_algebra_float_data_set_malloc ( const fds_selector s ) {

    fds *ds = ( fds * ) scalable_aligned_malloc ( sizeof ( fds ), 64 );

    if ( ds == nullptr ) {

        return nullptr;
    }

    // tr? = training?, va? = validation?, te? = test?...

    float *tri = nullptr, *tro = nullptr, *vai = nullptr, *vao = nullptr, *tei = nullptr, *teo = nullptr;

    // ds->fds_cppptr = fds_get ( s, &ds->n_iput, &ds->n_oput, &ds->training.n_patt, &tri, &tro, &ds->validation.n_patt, &vai, &vao, &ds->test.n_patt, &tei, &teo );

    if ( ds->fds_cppptr == nullptr ) {

        return nullptr;
    }

    ds->training.iput   = linear_algebra_float_malloc_memptr ( tri,   ds->training.n_patt, ds->n_iput + 1 );
    ds->training.oput   = linear_algebra_float_malloc_memptr ( tro,   ds->training.n_patt, ds->n_oput     );

    ds->validation.iput = linear_algebra_float_malloc_memptr ( vai, ds->validation.n_patt, ds->n_iput + 1 );
    ds->validation.oput = linear_algebra_float_malloc_memptr ( vao, ds->validation.n_patt, ds->n_oput     );

    ds->test.iput       = linear_algebra_float_malloc_memptr ( tei,       ds->test.n_patt, ds->n_iput + 1 );
    ds->test.oput       = linear_algebra_float_malloc_memptr ( teo,       ds->test.n_patt, ds->n_oput     );

    const std::uint32_t r = 3 * (   ds->training.iput == nullptr ) +  5 * (   ds->training.iput == nullptr )
                    +  7 * ( ds->validation.iput == nullptr ) + 11 * ( ds->validation.iput == nullptr )
                    + 13 * (       ds->test.iput == nullptr ) + 17 * (       ds->test.iput == nullptr );

    if ( r != 0 && r != 8 && r != 18 && r != 26 && r != 30 && r != 38 && r != 48 ) {

        // linear_algebra_float_data_set_free ( ds );

        return nullptr;
    }

    /*

    linear_algebra_float_print ( ds->training.iput, 3 );
    linear_algebra_float_print ( ds->training.oput, 3 );

    linear_algebra_float_print ( ds->validation.iput, 3 );
    linear_algebra_float_print ( ds->validation.oput, 3 );

    linear_algebra_float_print ( ds->test.iput, 3 );
    linear_algebra_float_print ( ds->test.oput, 3 );

    */

    return ds;
}
