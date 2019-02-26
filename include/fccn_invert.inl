
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


template<typename real, typename index, typename sfinae>
lapack_int fccn<real, index, sfinae>::levenberg_marquardt_lu_invert_hes ( ) {

    std::memcpy ( ( void* ) ihs_data, ( void* ) hes_data, wgts_sqr * sizeof ( real ) );

    lapack_int info = lapack_int { 0 };

    if constexpr ( std::is_same<value_type, float>::value ) {

        info = LAPACKE_sgetrf ( CblasColMajor, ( lapack_int ) wgts, ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );
       
        if ( not ( info ) ) {

            info = LAPACKE_sgetri ( CblasColMajor, ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );
        }
    }

    else {

        info = LAPACKE_dgetrf ( CblasColMajor, ( lapack_int ) wgts, ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );

        if ( not ( info ) ) {

            info = LAPACKE_dgetri ( CblasColMajor, ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );
        }
    }

    return info;
}


template<typename real, typename index, typename sfinae>
lapack_int fccn<real, index, sfinae>::levenberg_marquardt_cholesky_invert_hes ( ) {

    std::memcpy ( ( void* ) ihs_data, ( void* ) hes_data, wgts_sqr * sizeof ( real ) );

    lapack_int info = lapack_int { 0 };

    if constexpr ( std::is_same<value_type, float>::value ) {

        info = LAPACKE_spotrf ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts );

        if ( not ( info ) ) {

            info = LAPACKE_spotri ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts );
        }
    }

    else {

        info = LAPACKE_dpotrf ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts );

        if ( not ( info ) ) {

            info = LAPACKE_dpotri ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts );
        }
    }

    pointer t = ihs_data, l = t + wgts_sqr, f = ihs_data;

    for ( index i = index { 0 }, incr = wgts + index { 1 }; i < wgts; ++i, t += incr, f += i ) {

        for ( pointer p = t; p < l; p += wgts, ++f ) {

            *p = *f;
        }
    }

    return info;
}


template<typename real, typename index, typename sfinae>
lapack_int fccn<real, index, sfinae>::levenberg_marquardt_bunch_kaufman_invert_hes ( ) {

    std::memcpy ( ( void* ) ihs_data, ( void* ) hes_data, wgts_sqr * sizeof ( real ) );

    lapack_int info = lapack_int { 0 };

    if constexpr ( std::is_same<value_type, float>::value ) {

        info = LAPACKE_ssytrf ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );

        if ( not ( info ) ) {

            info = LAPACKE_ssytri ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );
        }
    }

    else {

        info = LAPACKE_dsytrf ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );

        if ( not ( info ) ) {

            info = LAPACKE_dsytri ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );
        }
    }

    pointer t = ihs_data, l = t + wgts_sqr, f = ihs_data;

    for ( index i = index { 0 }, incr = wgts + index { 1 }; i < wgts; ++i, t += incr, f += i ) {

        for ( pointer p = t; p < l; p += wgts, ++f ) {

            *p = *f;
        }
    }

    return info;
}


template<typename real, typename index, typename sfinae>
lapack_int fccn<real, index, sfinae>::levenberg_marquardt_bunch_kaufman_rook_invert_hes ( ) {

    std::memcpy ( ( void* ) ihs_data, ( void* ) hes_data, wgts_sqr * sizeof ( real ) );

    lapack_int info = lapack_int { 0 };

    if constexpr ( std::is_same<value_type, float>::value ) {

        info = LAPACKE_ssytrf_rook ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );

        if ( not ( info ) ) {

            info = LAPACKE_ssytri ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );
        }
    }

    else {

        info = LAPACKE_dsytrf_rook ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );

        if ( not ( info ) ) {

            info = LAPACKE_dsytri ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );
        }
    }

    pointer t = ihs_data, l = t + wgts_sqr, f = ihs_data;

    for ( index i = index { 0 }, incr = wgts + index { 1 }; i < wgts; ++i, t += incr, f += i ) {

        for ( pointer p = t; p < l; p += wgts, ++f ) {

            *p = *f;
        }
    }

    return info;
}


template<typename real, typename index, typename sfinae>
lapack_int fccn<real, index, sfinae>::levenberg_marquardt_aasen_invert_hes ( ) {

    std::memcpy ( ( void* ) ihs_data, ( void* ) hes_data, wgts_sqr * sizeof ( real ) );

    lapack_int info = lapack_int { 0 };

    if constexpr ( std::is_same<value_type, float>::value ) {

        info = LAPACKE_ssytrf_aa_work ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv, lapack_work, wgts * index { 64 } );

        if ( not ( info ) ) {

            info = LAPACKE_ssytri ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );
        }
    }

    else {

        info = LAPACKE_dsytrf_aa_work ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv, lapack_work, wgts * index { 64 } );

        if ( not ( info ) ) {

            info = LAPACKE_dsytri ( CblasColMajor, 'L', ( lapack_int ) wgts, ihs_data, ( lapack_int ) wgts, lapack_ipiv );
        }
    }

    pointer t = ihs_data, l = t + wgts_sqr, f = ihs_data;

    for ( index i = index { 0 }, incr = wgts + index { 1 }; i < wgts; ++i, t += incr, f += i ) {

        for ( pointer p = t; p < l; p += wgts, ++f ) {

            *p = *f;
        }
    }

    return info;
}
