
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

#include "fully_connected_ann.h"

#include <float.h>
#include <stdbool.h>


inline void fully_connected_cascade_nguyen_widrow_weights ( fmat *w, const std::uint32_t n_iput, const std::uint32_t n_oput ) {

    // Nguyen-Widrow weight initialization...

    const float beta = 0.7f * powf ( ( float ) n_oput, 1.0f / ( float ) ( n_iput +1 ) );

    linear_algebra_float_uniform ( w, -1.0f, 1.0f );

    for ( std::uint32_t c = 0; c < w->n_elem; ++c ) {

        w->v [ c ] *= beta / linear_algebra_float_norm ( w );
    }
}


fcn *fully_connected_cascade_malloc ( std::uint32_t n_nrns, const std::uint32_t n_iput, const std::uint32_t n_oput, const activation_function activation_function, const float alpha ) {

    n_nrns = MAX ( n_nrns, n_oput );

    fcn *r = ( fcn * ) scalable_aligned_malloc ( sizeof ( fcn ) + n_nrns * sizeof ( fmat * ), _ALIGN );

    r->n_nrns = n_nrns;
    r->n_iput = n_iput;
    r->n_oput = n_oput;

    const std::uint32_t n_iput_1 = n_iput + 1, n_wgts = ( n_nrns * ( n_nrns + n_iput + n_iput_1 ) ) / 2;

    r->n_wgts = n_wgts;

    r->ibo_o0 = n_iput_1 + n_nrns - n_oput;
    r->nbn_o0 = n_nrns - n_oput;

    r->alpha  = alpha;

    switch ( activation_function ) {

        case activation_function::unipolar:

            r->activation_function = activation_unipolar;
            r->activation_function_derivative = activation_unipolar_derivative;

            break;

        case activation_function::bipolar:

            r->activation_function = activation_bipolar;
            r->activation_function_derivative = activation_bipolar_derivative;

            break;

        case activation_function::rectifier:

            r->activation_function = activation_rectifier;
            r->activation_function_derivative = activation_rectifier_derivative;

            break;

        default:

            printf ( "undefined activation_function\n" );

            exit ( 0 );
    }

    r->ibo = nullptr;
    r->wts = linear_algebra_float_malloc ( n_wgts, 1 );

    #if 0
    fully_connected_cascade_nguyen_widrow_weights ( r->wts, r->n_iput, r->n_oput );
    #else
    linear_algebra_float_uniform ( r->wts, -1.0f, 1.0f );
    #endif

    float *wts_ptr = r->wts->v;

    for ( std::uint32_t i = 0; i < n_nrns; ++i ) {

        const std::uint32_t n_wts_in_layer = n_iput_1 + i;

        r->lyr [ i ] = linear_algebra_float_malloc_memptr ( wts_ptr, n_wts_in_layer, 1 );

        wts_ptr += n_wts_in_layer;
    }

    return r;
}


fcn *fully_connected_recurrent_cascade_malloc ( std::uint32_t n_nrns, const std::uint32_t n_iput, const std::uint32_t n_oput, const activation_function activation_function, const float alpha ) {

    return fully_connected_cascade_malloc ( n_nrns, n_iput + n_nrns, n_oput, activation_function, alpha );
}


fcn *fully_connected_cascade_copy ( fcn *nn ) {

    fcn *r = ( fcn * ) scalable_aligned_malloc ( sizeof ( fcn ) + nn->n_nrns * sizeof ( fmat * ), _ALIGN );

    r->n_nrns = nn->n_nrns;
    r->n_iput = nn->n_iput;
    r->n_oput = nn->n_oput;
    r->n_wgts = nn->n_wgts;
    r->ibo_o0 = nn->ibo_o0;
    r->nbn_o0 = nn->nbn_o0;

    r->alpha = nn->alpha;

    r->activation_function = nn->activation_function;
    r->activation_function_derivative = nn->activation_function_derivative;

    if ( nn->ibo != nullptr ) {

        r->ibo = linear_algebra_float_malloc ( nn->ibo->n_rows, nn->ibo->n_cols );
        linear_algebra_float_copy ( r->ibo, nn->ibo );
    }

    else {

        r->ibo = nullptr;
    }

    r->wts = linear_algebra_float_malloc ( nn->n_wgts, 1 );
    linear_algebra_float_copy ( r->wts, nn->wts );

    float *wts_ptr = r->wts->v;

    for ( std::uint32_t i = 0; i < nn->n_nrns; ++i ) {

        const std::uint32_t n_wts_in_layer = nn->n_iput + 1 + i;

        r->lyr [ i ] = linear_algebra_float_malloc_memptr ( wts_ptr, n_wts_in_layer, 1 );

        wts_ptr += n_wts_in_layer;
    }

    return r;
}


flm *fully_connected_cascade_levenberg_marquardt_malloc ( const std::uint32_t n_nrns, const std::uint32_t n_wgts ) {

    const std::uint32_t n_awts = ( ( n_wgts * ( n_wgts -1 ) ) / 2 ) + n_wgts;

    const std::uint32_t nn_padded = linear_algebra_float_padded ( n_nrns * n_nrns * sizeof ( float ), _ALIGN );
    const std::uint32_t w1_padded = linear_algebra_float_padded (          n_wgts * sizeof ( float ), _ALIGN );
    const std::uint32_t ac_padded = linear_algebra_float_padded (          n_awts * sizeof ( float ), _ALIGN );
    const std::uint32_t w2_padded = linear_algebra_float_padded ( n_wgts * n_wgts * sizeof ( float ), _ALIGN );

    flm *r = ( flm * ) scalable_aligned_malloc ( sizeof ( flm ) + ( nn_padded + w1_padded + ac_padded + w1_padded + w2_padded + w2_padded ), _ALIGN );

    r->lambda = 0.1f;
    r->v = 10.0f;

    r->lambda_min = FLT_EPSILON * r->v;
    r->lambda_max = FLT_MAX / r->v;

    r->bay_alpha = 0.0f;
    r->bay_beta = 1.0f;

    char *ptr = ( char * ) r->f;

    r->nbn = linear_algebra_float_malloc_memptr ( ( float * ) ( ptr              ), n_nrns, n_nrns );
    r->jpm = linear_algebra_float_malloc_memptr ( ( float * ) ( ptr += nn_padded ), n_wgts,      1 );
    r->acc = linear_algebra_float_malloc_memptr ( ( float * ) ( ptr += w1_padded ), n_awts,      1 );
    r->gra = linear_algebra_float_malloc_memptr ( ( float * ) ( ptr += ac_padded ), n_wgts,      1 );
    r->hes = linear_algebra_float_malloc_memptr ( ( float * ) ( ptr += w1_padded ), n_wgts, n_wgts );
    r->ihs = linear_algebra_float_malloc_memptr ( ( float * ) ( ptr += w2_padded ), n_wgts, n_wgts );

    return r;
}


float fully_connected_cascade_sse ( const sds *dt, const fmat *ibo ) {

    // Sum-of-squares error...

    float sse = 0.0f, *dv = dt->oput->v, *iv = ibo->v;

    const std::uint32_t dr = dt->oput->n_rows, dc = dt->oput->n_cols, id = ibo->n_cols - dt->oput->n_cols;

    const float *dvr = dv + dr;

    for ( std::uint32_t j = 0; j < dc; ++j ) {

        const float *dl = dvr + j * dr;

        for ( float *d = dv + j * dr, *i = iv + ( j + id ) * dr; d < dl; ++d, ++i ) {

            const float t = *d - *i;

            sse += t * t;
        }
    }

    return 0.5f * sse / ( float ) ( dr );
}


float fully_connected_cascade_ssw ( const fcn *nn ) {

    // Sum-of-squares weights...

    const std::uint32_t n_wgts = nn->n_wgts;
    const float *wts_v = nn->wts->v;

    float ssw = 0.0f;

    for ( std::uint32_t i = 0; i < n_wgts; ++i ) {

        const float w = wts_v [ i ];

        ssw += w * w;
    }

    return 0.5f * ssw / ( float ) n_wgts;
}


float fully_connected_cascade_cle ( const sds *dt, const fmat *ibo ) {

    // Classification error...

    std::uint32_t cle = 0;

    const std::uint32_t n_rows = dt->oput->n_rows, n_cols = dt->oput->n_cols;
    float *dv = dt->oput->v, *iv = ibo->v + ( ibo->n_cols - n_cols ) * n_rows;

    for ( std::uint32_t r = 0; r < n_rows; ++r ) {

        float *d = dv++, d_max = *d, *i = iv++, i_max = *i;
        std::uint32_t c = 0, d_idx = 0, i_idx = 0;

        while ( ++c < n_cols ) {

            if ( d += n_rows, *d > d_max ) {

                d_max = *d, d_idx = c;
            }

            if ( i += n_rows, *i > i_max ) {

                i_max = *i, i_idx = c;
            }
        }

        cle += d_idx != i_idx;
    }

    return ( float ) cle / ( float ) n_rows;
}


void fully_connected_cascade_accumulator_fill ( const fmat *v, const std::uint32_t n_elem, fmat *accumulator ) {

    // Accumulate auto vector outer product...

    const std::uint32_t v_n_elem_minus_n_elem = v->n_elem - n_elem;

    const float *vv = v->v;
    float *av = accumulator->v;

    for ( std::uint32_t i = 0, k = 0; i < n_elem; ++i, k += v_n_elem_minus_n_elem ) {

        for ( std::uint32_t j = i; j < n_elem; ++j, ++k ) {

            av [ k ] += vv [ i ] * vv [ j ];
        }
    }
}


void fully_connected_cascade_accumulator_unfold ( const fmat *accumulator, fmat *m ) {

    // Restore square matrix...

    const float *av = accumulator->v;

    for ( std::uint32_t i = 0, k = 0; i < ( const std::uint32_t ) m->n_cols; ++i, ++k ) {

        m->at ( i, i ) = av [ k ];

        for ( std::uint32_t j = i + 1; j < ( const std::uint32_t ) m->n_rows; j++ ) {

            m->at ( i, j ) = m->at ( j, i ) = av [ ++k ];
        }
    }
}


inline float fully_connected_cascade_hessian_gradients_dot ( const float *v1, const float *v2, const std::uint32_t k ) {

    float dot = 0.0f;

    for ( float *v1i = ( float * ) v1, *v2i = ( float * ) v2; v1i < ( v1 + k ); ) {

        dot += *v1i++ * *v2i++;
    }

    return dot;
}

void fully_connected_cascade_hessian_gradients_old ( const fcn *nn, flm *lm, const sds *ds ) {

    const fmat *ibo = nn->ibo;

    fmat *nbn = lm->nbn, *jpm = lm->jpm, *acc = lm->acc, *gra = lm->gra, *hes = lm->hes;

    // Initiate...

    linear_algebra_float_zeros ( acc );
    linear_algebra_float_zeros ( nbn );
    linear_algebra_float_zeros ( gra );

    const std::uint32_t nn_n_nrns = nn->n_nrns, nn_n_iput = nn->n_iput, nn_n_oput = nn->n_oput, nn_n_iput_1 = nn_n_iput + 1;

    const std::uint32_t ibo_n_rows = ibo->n_rows, ibo_o0 = nn->ibo_o0;
    const std::uint32_t nbn_n_rows = nbn->n_rows, nbn_o0 = nn->nbn_o0;

    const float *data_oput_v = ds->oput->v, *ibo_v = ibo->v;

    float *jpm_v = jpm->v, *nbn_v = nbn->v, *gra_v = gra->v, *nbn_v_nbn_o0 = nbn_v + nbn_o0;

    float ( *nn_activation_function_derivative ) ( const float, const float ) = nn->activation_function_derivative;
    const float alpha = nn->alpha;

    // Fill in the weights...

    for ( std::uint32_t i = 1; i < nn_n_nrns; i++ ) {

        memcpy ( nbn_v + i * nbn_n_rows, nn->lyr [ i ]->v + nn_n_iput_1, ( nn->lyr [ i ]->n_elem - nn_n_iput_1 ) * sizeof ( float ) );
    }

    // Start calculation of hessian matrix and gradient vector...

    for ( std::uint32_t p = 0; p < ds->n_patt; p++ ) {

        const float *ibo_v_p = ibo_v + p;

        // Fill gain matrix...

        for ( std::uint32_t i = 0; i < nbn_n_rows; i++ ) {

            nbn_v [ i * nbn_n_rows + i ] = nn_activation_function_derivative ( ibo_v_p [ ( i + nn_n_iput_1 ) * ibo_n_rows ], alpha );
        }

        for ( std::uint32_t k = 1; k < nbn_n_rows; k++ ) {

            for ( std::uint32_t j = 0; j < k; j++ ) {

                nbn_v [ j * nbn_n_rows + k ] = nbn_v [ k * nbn_n_rows + k ] * fully_connected_cascade_hessian_gradients_dot ( nbn_v + j * nbn_n_rows + j, nbn_v + k * nbn_n_rows + j, k - j );
            }
        }

        // For every output neuron, only the non-zero elements...

        for ( std::uint32_t o = 0; o < nn_n_oput; o++ ) {

            std::uint32_t w = 0;

            // A line in jacobian...

            for ( std::uint32_t i = 0; i <= nbn_o0 + o; i++ ) {

                for ( std::uint32_t j = 0; j < nn_n_iput_1 + i; j++, w++ ) {

                    jpm_v [ w ] = -ibo_v [ j * ibo_n_rows + p ] * nbn_v_nbn_o0 [ i * nbn_n_rows + o ];
                }
            }

            // Accumulate vector outer product of a line in jacobian, w = number of non-zero elements...

            fully_connected_cascade_accumulator_fill ( jpm, w, acc );

            const float e = data_oput_v [ o * nbn_n_rows + p ] - ibo_v [ ( ibo_o0 + o ) * nbn_n_rows + p ];

            // Accumulate gradient...

            for ( std::uint32_t r = 0; r < w; r++ ) {

                gra_v [ r ] += jpm_v [ r ] * e;
            }
        }
    }

    fully_connected_cascade_accumulator_unfold ( acc, hes );
}


void fully_connected_cascade_hessian_gradients ( const fcn *nn, flm *lm, const sds *ds ) {

    // Optimised version 9% faster...

    const fmat *ibo = nn->ibo;

    fmat *nbn = lm->nbn, *jpm = lm->jpm, *acc = lm->acc, *gra = lm->gra, *hes = lm->hes, *ihs = lm->ihs;

    // Initiate...

    linear_algebra_float_zeros ( nbn );
    linear_algebra_float_zeros ( acc );
    linear_algebra_float_zeros ( gra );

    const std::uint32_t nn_n_nrns = nn->n_nrns, nn_n_iput = nn->n_iput, nn_n_oput = nn->n_oput, nn_n_iput_1 = nn_n_iput +1;

    const std::uint32_t ibo_n_rows = ibo->n_rows, ibo_o0 = nn->ibo_o0;
    const std::uint32_t nbn_n_rows = nbn->n_rows, nbn_o0 = nn->nbn_o0;

    const float *data_oput_v = ds->oput->v, *ibo_v = ibo->v;

    float *jpm_v = jpm->v, *nbn_v = nbn->v, *gra_v = gra->v, *nbn_v_nbn_o0 = nbn_v + nbn_o0;

    float ( *nn_activation_function_derivative ) ( const float, const float ) = nn->activation_function_derivative;
    const float alpha = nn->alpha;

    // Fill in the weights...

    for ( std::uint32_t i = 1; i < nn_n_nrns; i++ ) {

        const fmat *nn_lyr_i = nn->lyr [ i ];

        memcpy ( nbn_v + i * nbn_n_rows, nn_lyr_i->v + nn_n_iput_1, ( nn_lyr_i->n_elem - nn_n_iput_1 ) * sizeof ( float ) );
    }

    // Start calculation of hessian matrix and gradient vector...

    for ( std::uint32_t p = 0; p < ds->n_patt; p++ ) {

        const float *ibo_v_p                                          = ibo_v + p;
        const float *ibo_v_p_ibo_o0_ibo_n_rows                        = ibo_v_p + ibo_o0 * ibo_n_rows;
        const float *ibo_v_p_nn_n_iput_1_ibo_n_rows                   = ibo_v_p + nn_n_iput_1 * ibo_n_rows;
        const float *ibo_v_p_nn_n_iput_1_ibo_n_rows_nbn_o0_ibo_n_rows = ibo_v_p_nn_n_iput_1_ibo_n_rows + nbn_o0 * ibo_n_rows;

        const float *data_oput_v_p                                    = data_oput_v + p;

        // Fill gain matrix...

        for ( std::uint32_t i = 0; i < nbn_n_rows; i++ ) {

            nbn_v [ i * nbn_n_rows + i ] = nn_activation_function_derivative ( ibo_v_p_nn_n_iput_1_ibo_n_rows [ i * ibo_n_rows ], alpha );
        }

        for ( std::uint32_t k = 1; k < nbn_n_rows; k++ ) {

            float *nbn_v_k = nbn_v + k;

            for ( std::uint32_t j = 0; j < k; j++ ) {

                nbn_v_k [ j * nbn_n_rows ] = nbn_v_k [ k * nbn_n_rows ] * fully_connected_cascade_hessian_gradients_dot ( nbn_v + j * nbn_n_rows + j, nbn_v + k * nbn_n_rows + j, k - j );
            }
        }

        // For every output neuron, only the non-zero elements...

        for ( std::uint32_t o = 0, o_ibo_n_rows = 0; o < nn_n_oput; o++, o_ibo_n_rows += ibo_n_rows ) {

            std::uint32_t w = 0;

            const float *nbn_v_nbn_o0_o = nbn_v_nbn_o0 + o, *h = ibo_v_p_nn_n_iput_1_ibo_n_rows_nbn_o0_ibo_n_rows + o_ibo_n_rows;

            // A line in Jacobian...

            for ( float *i = ( float * ) ibo_v_p_nn_n_iput_1_ibo_n_rows; i <= h; i += ibo_n_rows, nbn_v_nbn_o0_o += nbn_n_rows ) {

                for ( float *j = ( float * ) ibo_v_p; j < i; j += ibo_n_rows, w++ ) {

                    jpm_v [ w ] = -*j * *nbn_v_nbn_o0_o;
                }
            }

            // Accumulate vector outer product of a line in Jacobian, w = number of non-zero elements...

            fully_connected_cascade_accumulator_fill ( jpm, w, acc );

            const float e = data_oput_v_p [ o_ibo_n_rows ] - ibo_v_p_ibo_o0_ibo_n_rows [ o_ibo_n_rows ];

            // Accumulate gradient...

            for ( std::uint32_t r = 0; r < w; r++ ) {

                gra_v [ r ] += jpm_v [ r ] * e;
            }
        }
    }

    fully_connected_cascade_accumulator_unfold ( acc, hes );
}

#include <sax/autotimer.hpp>

int i = 0;

float fully_connected_cascade_levenberg_marquardt_epoch ( fcn *nn, flm *lm, const sds *ds ) {

    fmat *ibo = nn->ibo;

    fully_connected_cascade_feedforward ( nn );
    fully_connected_cascade_hessian_gradients ( nn, lm, ds );

    fmat *dlt = linear_algebra_float_malloc ( nn->n_wgts, 1 ); // Column vector...
    fmat *dia = linear_algebra_float_malloc ( nn->n_wgts, 1 ); // Column vector...

    bool delta_is_not_empty = false;

    linear_algebra_float_copy_dia ( dia, lm->hes );

    const float objective = fully_connected_cascade_sse ( ds, ibo );
    float sse = FLT_MAX;

    // Begin of the main Levenberg-Marquardt method...

    do {

        if ( delta_is_not_empty ) {

            linear_algebra_float_add ( nn->wts, dlt );
        }

        linear_algebra_float_dia_copy_add ( lm->hes, dia, lm->lambda );

        // std::cout << lm->lambda << '\n';
        // linear_algebra_float_print ( lm->hes, 3 );

        const lapack_int info = linear_algebra_float_invert ( lm->ihs, lm->hes );

        if ( lapack_int { 0 } != info ) {

            // The Hessian is singular, continue to the next
            // iteration until the diagonal update transforms
            // it back to non-singular...

            lm->lambda *= lm->v;

            if ( lm->lambda < lm->lambda_max ) {

                delta_is_not_empty = false;

                continue;
            }
        }

        linear_algebra_float_mul_mv ( dlt, lm->ihs, lm->gra );
        linear_algebra_float_sub ( nn->wts, dlt );
        delta_is_not_empty = true;

        // linear_algebra_float_print_transposed ( dlt, 3 );

        fully_connected_cascade_feedforward ( nn );
        sse = fully_connected_cascade_sse ( ds, ibo );

        lm->lambda *= lm->v;

    } while ( sse >= objective && lm->lambda < lm->lambda_max );

    // If this iteration caused a error drop, then the
    // next iteration will use a smaller damping factor...

    if ( sse < objective ) {

        lm->lambda /= lm->v;
    }

    linear_algebra_float_free ( dia );
    linear_algebra_float_free ( dlt );

    return sse;
}


float fully_connected_cascade_levenberg_marquardt_baysian_epoch ( fcn *nn, flm *lm, const sds *ds ) {

    fmat *ibo = nn->ibo;

    fully_connected_cascade_feedforward ( nn );
    fully_connected_cascade_hessian_gradients ( nn, lm, ds );

    fmat *dlt = linear_algebra_float_malloc ( nn->n_wgts, 1 ); // Column vector...
    fmat *dia = linear_algebra_float_malloc ( nn->n_wgts, 1 ); // Column vector...

    bool delta_is_not_empty = false;

    linear_algebra_float_copy_dia ( dia, lm->hes );

    const float objective = lm->bay_beta * fully_connected_cascade_sse ( ds, ibo ) + lm->bay_alpha * fully_connected_cascade_ssw ( nn );
    float sse = FLT_MAX, ssw = FLT_MAX;

    // Begin of the main Levenberg-Marquardt method...

    do {

        if ( delta_is_not_empty ) {

            linear_algebra_float_add ( nn->wts, dlt );
        }

        linear_algebra_float_dia_copy_add ( lm->hes, dia, lm->lambda + lm->bay_alpha );

        if ( linear_algebra_float_invert ( lm->ihs, lm->hes ) != 0 ) {

            // The Hessian is singular, continue to the next
            // iteration until the diagonal update transforms
            // it back to non-singular...

            lm->lambda *= lm->v;

            if ( lm->lambda < lm->lambda_max ) {

                delta_is_not_empty = false;

                continue;
            }

            else {

                break;
            }
        }

        linear_algebra_float_mul_mv ( dlt, lm->ihs, lm->gra );

        delta_is_not_empty = true;

        linear_algebra_float_sub ( nn->wts, dlt );

        fully_connected_cascade_feedforward ( nn );

        sse = fully_connected_cascade_sse ( ds, ibo );
        ssw = fully_connected_cascade_ssw ( nn );

        lm->lambda *= lm->v;

    } while ( ( lm->bay_beta * sse + lm->bay_alpha * ssw ) >= objective && lm->lambda < lm->lambda_max );

    // If this iteration caused a error drop, then the
    // next iteration will use a smaller damping factor...

    lm->lambda /= lm->v;

    const float trace_inv_hes = linear_algebra_float_trace ( lm->ihs );

    // Poland's update formula...

    lm->bay_beta = ( ds->n_patt - ( nn->n_wgts - lm->bay_alpha * trace_inv_hes ) ) / ( 2.0f * sse );
    lm->bay_alpha = nn->n_wgts / ( 2.0f * ssw + trace_inv_hes );

    linear_algebra_float_free ( dia );
    linear_algebra_float_free ( dlt );

    return sse;
}


float fully_connected_cascade_train ( fcn *nn, const fds *ds ) {

    fully_connected_cascade_io_malloc ( nn, &ds->training );

    flm *lm = fully_connected_cascade_levenberg_marquardt_malloc ( nn->n_nrns, nn->n_wgts );

    float new_sse = FLT_MAX, old_sse = 0.0f;

    std::uint32_t c = 0;

    do {

        // if ( old_sse == new_sse ) { --c; }

        old_sse = new_sse;
        new_sse = fully_connected_cascade_levenberg_marquardt_epoch ( nn, lm, &ds->training );

        std::printf ( "%3u - sse: %.8f\n", c++, new_sse );

        // linear_algebra_float_mul ( nn->wts, 0.99f ); // Decay...

    } while ( new_sse >= 0.01f );

    // fully_connected_cascade_print_errs ( nn, lm, &ds->training, "training" );

    fully_connected_cascade_levenberg_marquardt_free ( lm );
    fully_connected_cascade_io_free ( nn );

    return new_sse;
}


float fully_connected_cascade_baysian_train ( fcn *nn, const fds *ds ) {

    fully_connected_cascade_io_malloc ( nn, &ds->training );

    flm *lm = fully_connected_cascade_levenberg_marquardt_malloc ( nn->n_nrns, nn->n_wgts );

    float new_sse = FLT_MAX, old_sse = 0.0f;

    std::uint32_t c = 0;

    do {

        if ( old_sse == new_sse ) { --c; }

        old_sse = new_sse;
        new_sse = fully_connected_cascade_levenberg_marquardt_baysian_epoch ( nn, lm, &ds->training );

        printf ( "%3u - sse: %.8f\n", c++, new_sse );

        // linear_algebra_float_mul ( nn->wts, 0.99f ); // Decay...

    } while ( new_sse != old_sse );

    fully_connected_cascade_print_errs ( nn, lm, &ds->training, "training" );

    fully_connected_cascade_levenberg_marquardt_free ( lm );
    fully_connected_cascade_io_free ( nn );

    return new_sse;
}



void fully_connected_cascade_backpropagate ( flm *, const sds * ) { // 4, 7, 3
//	void fully_connected_cascade_backpropagate ( flm *lm, const sds *ss ) { // 4, 7, 3
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


void fully_connected_cascade_print_errs ( fcn *nn_, flm *lm, const sds *ss, const char *s ) {

    fcn *nn = fully_connected_cascade_copy ( nn_ ); // So it's useable with recurrent networks...

    fully_connected_cascade_feedforward ( nn );

    const float new_sse = fully_connected_cascade_sse ( ss, nn->ibo ), new_ssw = fully_connected_cascade_ssw ( nn );

    printf ( " obj_val %s: %.8f\n", s, lm->bay_beta * new_sse + lm->bay_alpha * new_ssw );
    printf ( " sse_err %s: %.8f\n", s, new_sse );
    printf ( " ssw_val %s: %.8f\n", s, new_ssw );
    printf ( " cls_err %s: %.8f\n\n", s , fully_connected_cascade_cle ( ss, nn->ibo ) );

    fully_connected_cascade_free ( nn );
}


void fully_connected_cascade_print_wgts ( const fcn *nn ) {

    fmat *w = linear_algebra_float_calloc ( nn->n_nrns + nn->n_iput, nn->n_nrns );

    for ( std::uint32_t i = 0; i < ( const std::uint32_t ) nn->n_nrns; ++i ) {

        memcpy ( w->v + i * w->n_rows, nn->lyr [ i ]->v, nn->lyr [ i ]->n_elem * sizeof ( float ) );
    }

    linear_algebra_float_print ( w, 3 );
    linear_algebra_float_free ( w );
}
