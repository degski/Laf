
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
fccn_basic<real, index, sfinae>::fccn_basic ( index nrns_, const index iput_, const index oput_, const activation_function activation_function_, const real alpha_ ) {
    nrns = std::max ( nrns_, oput_ );
    iput = iput_;
    oput = oput_;
    const index iput_1 = iput + index { 1 };
    cols = iput_1 + nrns;
    wgts = ( nrns * ( cols + iput ) ) / index { 2 };
    wgts_sqr = wgts * wgts;
    switch ( activation_function_ ) {
    case activation_function::unipolar:
        activation = activation_unipolar;
        break;
    case activation_function::bipolar:
        activation = activation_bipolar;
        break;
    case activation_function::rectifier:
        activation = activation_rectifier;
        break;
    default:
        std::printf ( "undefined activation_function\n" );
        std::abort ( );
    }
    alpha = alpha_;
    wts_data = ( pointer ) scalable_aligned_malloc ( wgts * sizeof ( value_type ), _ALIGN );
}


template<typename real, typename index, typename sfinae>
fccn_basic<real, index, sfinae>::~fccn_basic ( ) {
    scalable_aligned_free ( wts_data );
    matrix::destroy ( tr_ibo );
}


template<typename real, typename index, typename sfinae>
typename fccn_basic<real, index, sfinae>::fccn_base_ptr fccn_basic<real, index, sfinae>::construct ( index nrns_, const index iput_, const index oput_, const activation_function activation_function_, const real alpha_ ) {
    fccn_base_ptr n = ( fccn_base_ptr ) scalable_aligned_malloc ( fccn_base_size ( ), matrix::alignment ( ) );
    new ( n ) fccn_basic ( nrns_, iput_, oput_, activation_function_, alpha_ );
    return n;
}


template<typename real, typename index, typename sfinae>
void fccn_basic<real, index, sfinae>::feedforward ( ) noexcept {
    index c = iput + index { 1 };
    pointer x = wts_data, y = tr_ibo_data + tr_patt * c, e = y + tr_patt;
    while ( c < cols ) {
        if constexpr ( std::is_same<value_type, float>::value ) {
            cblas_sgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) tr_patt, ( lapack_int ) c, 1.0f, tr_ibo_data, ( lapack_int ) tr_patt, x, lapack_int { 1 }, 0.0f, y, lapack_int { 1 } );
        }
        else {
            cblas_dgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) tr_patt, ( lapack_int ) c, 1.0, tr_ibo_data, ( lapack_int ) tr_patt, x, lapack_int { 1 }, 0.0, y, lapack_int { 1 } );
        }
        while ( y < e ) {
            *y = activation ( *y, alpha );
            ++y;
        } // Apply activation function.
        x += c++; e += tr_patt;
    }
}


template<typename real, typename index, typename sfinae>
typename fccn_basic<real, index, sfinae>::pointer fccn_basic<real, index, sfinae>::convert ( const fmat *m_ ) {
    const index l = m_->n_rows * m_->n_cols;
    pointer t = ( pointer ) scalable_aligned_malloc ( l * sizeof ( value_type ), _ALIGN );
    auto f = m_->v;
    for ( index i = index { 0 }; i < l; ++i ) {
        t [ i ] = ( value_type ) f [ i ];
    }
    return t;
}


template<typename real, typename index, typename sfinae>
fccn<real, index, sfinae>::fccn ( index nrns_, const index iput_, const index oput_, const activation_function activation_function_, const real alpha_ ) {
    nrns = std::max ( nrns_, oput_ );
    iput = iput_;
    oput = oput_;
    const index iput_1 = iput + index { 1 };
    cols = iput_1 + nrns;
    wgts = ( nrns * ( cols + iput ) ) / index { 2 };
    wgts_sqr = wgts * wgts;
    switch ( activation_function_ ) {
    case activation_function::unipolar:
        activation = activation_unipolar;
        activation_derivative = activation_unipolar_derivative;
        break;
    case activation_function::bipolar:
        activation = activation_bipolar;
        activation_derivative = activation_bipolar_derivative;
        break;
    case activation_function::rectifier:
        activation = activation_rectifier;
        activation_derivative = activation_rectifier_derivative;
        break;
    default:
        std::printf ( "undefined activation_function\n" );
        std::abort ( );
    }
    alpha = alpha_;
    // Align second half of wts.
    index quo = wgts / ( 64 / sizeof ( real ) );
    quo += ( wgts - quo * ( 64 / sizeof ( real ) ) ) > index { 0 };
    quo *= ( 64 / sizeof ( real ) );
    wts_data = nguyen_widrow_weights ( ( pointer ) scalable_aligned_malloc ( quo * ( index { 2 } * sizeof ( value_type ) ), _ALIGN ) );
    wts_pdata = wts_data + quo;
    jpm_data = ( pointer ) scalable_aligned_malloc ( wgts * sizeof ( value_type ), _ALIGN );
    jwc_data = construct_jpm_wts_count ( );
    nwc_data = construct_nrns_wts_count ( );
    gra_data = ( pointer ) scalable_aligned_malloc ( wgts * sizeof ( value_type ), _ALIGN );
    dlt_data = ( pointer ) scalable_aligned_malloc ( quo * ( index { 2 } * sizeof ( value_type ) ), _ALIGN );
    std::memset ( ( void* ) dlt_data, 0, quo * ( index { 2 } * sizeof ( value_type ) ) );
    dlt_pdata = dlt_data + quo;
    dia_data = ( pointer ) scalable_aligned_malloc ( wgts * sizeof ( value_type ), _ALIGN );
    nbn = matrix::construct ( nrns, nrns );
    nbn_data = nbn->data ( );
    hes = matrix::construct ( wgts, wgts );
    hes_data = hes->data ( );
    ihs = matrix::construct ( wgts, wgts );
    ihs_data = ihs->data ( );
    lapack_ipiv = ( lapack_int* ) scalable_aligned_malloc ( wgts * sizeof ( lapack_int ), _ALIGN );
    lapack_work = ( pointer ) scalable_aligned_malloc ( wgts * index { 64 } * sizeof ( real ), _ALIGN );
}


template<typename real, typename index, typename sfinae>
fccn<real, index, sfinae>::~fccn ( ) {
    scalable_aligned_free ( wts_data < wts_pdata ? wts_data : wts_pdata );
    scalable_aligned_free ( jpm_data );
    scalable_aligned_free ( jwc_data );
    scalable_aligned_free ( nwc_data );
    scalable_aligned_free ( gra_data );
    scalable_aligned_free ( dlt_data < dlt_pdata ? dlt_data : dlt_pdata );
    scalable_aligned_free ( dia_data );
    matrix::destroy ( tr_ibo );
    matrix::destroy ( tr_out );
    matrix::destroy ( te_ibo );
    matrix::destroy ( te_out );
    matrix::destroy ( nbn );
    matrix::destroy ( hes );
    matrix::destroy ( ihs );
    scalable_aligned_free ( lapack_ipiv );
    scalable_aligned_free ( lapack_work );
    if ( nullptr != pm_data ) scalable_aligned_free ( pm_data );
    if ( nullptr != po_data ) scalable_aligned_free ( po_data );
    if ( nullptr != pop_data ) scalable_aligned_free ( pop_data );
    if ( nullptr != srt_data ) scalable_aligned_free ( srt_data );
}


template<typename real, typename index, typename sfinae>
index fccn<real, index, sfinae>::no_wgts ( index nrns_, const index iput_, const index oput_ ) noexcept {
    nrns_ = std::max ( nrns_, oput_ );
    return ( nrns_ * ( index { 2 } * iput_ + index { 1 } + nrns_ ) ) / index { 2 };
}


template<typename real, typename index, typename sfinae>
typename fccn<real, index, sfinae>::fccn_ptr fccn<real, index, sfinae>::construct ( index nrns_, const index iput_, const index oput_, const activation_function activation_function_, const real alpha_ ) {
    fccn_ptr n = ( fccn_ptr ) scalable_aligned_malloc ( fccn_size ( ), matrix::alignment ( ) );
    new ( n ) fccn ( nrns_, iput_, oput_, activation_function_, alpha_ );
    return n;
}


template<typename real, typename index, typename sfinae>
template<typename T>
void fccn<real, index, sfinae>::read ( T & col_data_, pointer f_, pointer i_, pointer o_, const index patt_, const index stride_, const activation_function activation_function_ ) noexcept {
    for ( const auto & cols : col_data_ ) {
        pointer & t = in_out::in == cols.first ? i_ : o_;
        std::memcpy ( ( void* ) t, ( void* ) f_, sizeof ( real ) * patt_ );
        if ( index { 1 } < cols.second ) {
            for ( pointer u = t, l = u + patt_, a = u; u < l; ++u, a = u ) {
                for ( auto code = to_gray ( ( index ) *a, cols.second ); code.size ( ); code.pop_back ( ), a += patt_ ) {
                    *a = activation_function_ == activation_function::unipolar ? ( real ) code.back ( ) * real { 1 } / real { 2 } : ( real ) ( ( int ) code.back ( ) - 1 );
                }
            }
        }
        t += cols.second * patt_;
        f_ += stride_;
    }
    std::fill ( i_, i_ + patt_, real { 1 } );
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::read_in_out ( const std::string & file_name_, const activation_function activation_function_ ) {
    std::ifstream file ( file_name_, std::ios_base::binary | std::ios_base::in );
    if ( file.is_open ( ) ) {
        csv_reader<real, index> reader ( file );
        matrix m ( reader.dimensions ( ), matrix_creation::zero );
        const index ll = reader.lines ( ) - index { 1 };
        std::vector<index> line_indices ( ll );
        std::iota ( std::begin ( line_indices ), std::end ( line_indices ), index { 0 } );
        std::shuffle ( std::begin ( line_indices ), std::end ( line_indices ), rng );
        for ( index l = index { 0 }; l < ll; ++l ) {
            reader.read_line ( );
            for ( index i = index { 0 }, j = index { 0 }; j < reader.length ( ); ++i ) {
                m.at ( line_indices [ l ], i ) = reader [ j++ ];
            }
        }
        const auto [ col_data, i_col_size, o_col_size ] = reader.columns ( );
        const index stride = m.rows;
        tr_patt = m.rows * real { 2 } / real { 3 }; // Set tr_patt.
        tr_ibo = matrix::construct_zero ( tr_patt, cols );
        tr_ibo_data = tr_ibo->data ( );
        tr_out = matrix::construct_zero ( tr_patt, o_col_size );
        tr_out_data = tr_out->data ( );
        read ( col_data, m.data ( ), tr_ibo_data, tr_out_data, tr_patt, stride, activation_function_ );
        te_patt = m.rows - tr_patt;
        te_ibo = matrix::construct_zero ( te_patt, cols );
        te_ibo_data = te_ibo->data ( );
        te_out = matrix::construct_zero ( te_patt, o_col_size );
        te_out_data = te_out->data ( );
        read ( col_data, m.data ( ) + tr_patt, te_ibo_data, te_out_data, te_patt, stride, activation_function_ );

        /*

        std::cout << tr_ibo << nl;
        std::cout << tr_out << nl;

        std::cout << te_ibo << nl;
        std::cout << te_out << nl;

        */

        file.close ( );
    }
    construct_pat ( );
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::construct_pop ( const index chro_, const index prnt_ ) {
    chro = chro_, prnt = prnt_;
    const index w1 = wgts + index { 1 };
    pop_data = ( pointer ) scalable_aligned_malloc ( chro * w1 * sizeof ( value_type ), _ALIGN );
    srt_data = ( pointer_ptr ) scalable_aligned_malloc ( chro * sizeof ( pointer ), _ALIGN );
    pointer_ptr s = srt_data;
    for ( pointer t = pop_data, l = t + chro * w1; t < l; t += w1, ++s ) {
        nguyen_widrow_weights ( t );
        *s = t;
    }
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::mutate ( pointer c_, pointer p_ ) noexcept {
    static auto noise = std::normal_distribution<real> ( real { 0 }, real { 0.05f } );
    real distance = real { 0 };
    for ( const pointer lc = c_ + wgts; c_ < lc; ++c_, ++p_ ) {
        const real n = noise ( rng );
        *c_ = *p_ + n; // Add gaussian noise to weights and assign to child.
        distance += n * n;
    }
    return std::sqrt ( distance );
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::mutate2 ( pointer c_, pointer p_ ) noexcept {
    constexpr index smpl = index { 32 }; const index w1 = wgts + index { 1 };
    static pointer can_data = ( pointer ) scalable_aligned_malloc ( w1 * ( smpl * sizeof ( value_type ) ), _ALIGN ); // can(didates).
    static pointer_ptr cst_data = ( pointer_ptr ) scalable_aligned_malloc ( smpl * sizeof ( pointer ), _ALIGN ); // sort
    static bool init = true;
    if ( init ) {
        pointer_ptr s = cst_data;
        for ( pointer t = can_data, l = t + smpl * w1; t < l; t += w1, ++s ) {
            *s = t;
        }
        init = false;
    }
    for ( pointer c = can_data, l = c + smpl * w1; c < l; c += w1 ) {
        c [ wgts ] = mutate ( c, p_ );
    }
    std::sort ( cst_data, cst_data + smpl, [ this ] ( const pointer & a, const pointer & b ) { return a [ wgts ] > b [ wgts ]; } );
    for ( pointer f = *cst_data, l = f + wgts; f < l; ++c_, ++f ) {
        *c_ = *f;
    }
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::crossover ( pointer c0_, pointer c1_, pointer p0_, pointer p1_ ) noexcept {
    static auto dis = std::uniform_int_distribution<index> ( index { 1 }, nrns - index { 2 } );
    const index w = nwc_data [ dis ( rng ) ];
    std::memcpy ( ( void* ) c0_, ( void* ) p0_, w * sizeof ( real ) );
    std::memcpy ( ( void* ) ( c0_ + w ), ( void* ) ( p1_ + w ), ( wgts - w ) * sizeof ( real ) );
    mutate ( c0_, c0_ );
    std::memcpy ( ( void* ) c1_, ( void* ) p1_, w * sizeof ( real ) );
    std::memcpy ( ( void* ) ( c1_ + w ), ( void* ) ( p0_ + w ), ( wgts - w ) * sizeof ( real ) );
    mutate ( c1_, c1_ );
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::regenerate_pop ( ) noexcept {
    const index w1 = wgts + index { 1 };
    for ( pointer f = pop_data, lf = f + chro * w1; f < lf; f += w1 ) {
        std::memcpy ( ( void* ) wts_data, ( void* ) f, wgts * sizeof ( real ) );
        f [ wgts ] = levenberg_marquardt_train1 ( index { 6 } ); // set the ase related to these weights.
        std::memcpy ( ( void* ) f, ( void* ) wts_data, wgts * sizeof ( real ) );
    }
    pointer_ptr l = srt_data + chro;
    std::sort ( srt_data, l, [ this ] ( const pointer & a, const pointer & b ) { return a [ wgts ] < b [ wgts ]; } );
    ska_sort ( srt_data + tr_patt, l ); // Sort the child pointers (by value), so access is better.
    --l;
    static auto select_parent = std::uniform_int_distribution<index> ( index { 0 }, prnt );
    static auto mutate_or_crossover = std::bernoulli_distribution ( real { 5 } / real { 10 } );
    for ( pointer_ptr parent = srt_data + select_parent ( rng ), child = srt_data + prnt; child < l; parent = srt_data + select_parent ( rng ), ++child ) {
        if ( mutate_or_crossover ( rng ) ) { // Mutate.
            mutate ( *child, *parent );
        }
        else { // Crossover.
            pointer_ptr spouse, sibling = child + index { 1 };
            while ( ( spouse = srt_data + select_parent ( rng ) ) == parent ); // Pick spouse different from parent to mate.
            crossover ( *child, *sibling, *parent, *spouse );
            ++child;
        }
    }
    std::memcpy ( ( void* ) ( *l ), ( void* ) ( *srt_data ), w1 * sizeof ( real ) ); // Elitisme.
    // nguyen_widrow_weights ( *l );
    // matrix m ( pop_data, wgts, chro );
    // std::ctr_out << m << nl;

    return ( *srt_data ) [ wgts ];
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::activate ( const pointer y_ ) noexcept {
    *y_ = activation ( *y_, alpha );
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::feedforward ( ) noexcept {
    index c = iput + index { 1 };
    pointer x = wts_data, y = tr_ibo_data + tr_patt * c, e = y + tr_patt;
    while ( c < cols ) {
        if constexpr ( std::is_same<value_type, float>::value ) {
            cblas_sgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) tr_patt, ( lapack_int ) c, 1.0f, tr_ibo_data, ( lapack_int ) tr_patt, x, lapack_int { 1 }, 0.0f, y, lapack_int { 1 } );
        }
        else {
            cblas_dgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) tr_patt, ( lapack_int ) c, 1.0, tr_ibo_data, ( lapack_int ) tr_patt, x, lapack_int { 1 }, 0.0, y, lapack_int { 1 } );
        }
        for ( ; y < e; activate ( y ), ++y );
        x += c++; e += tr_patt;
    }
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::feedforward_ase ( pointer x_ ) noexcept {
    index c = iput + index { 1 }, cnt = index { 0 };
    const index tr_out_col = c + nrns - oput;
    pointer y = tr_ibo_data + tr_patt * c, e = y + tr_patt, z = tr_out_data;
    real ase = real { 0 };
    while ( c < cols ) {
        if constexpr ( std::is_same<value_type, float>::value ) {
            cblas_sgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) tr_patt, ( lapack_int ) c, 1.0f, tr_ibo_data, ( lapack_int ) tr_patt, x_, lapack_int { 1 }, 0.0f, y, lapack_int { 1 } );
        }
        else {
            cblas_dgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) tr_patt, ( lapack_int ) c, 1.0, tr_ibo_data, ( lapack_int ) tr_patt, x_, lapack_int { 1 }, 0.0, y, lapack_int { 1 } );
        }
        if ( c < tr_out_col ) {
            while ( y < e ) {
                *y = activation ( *y, alpha ); // Apply activation function.
                ++y;
            }
        }
        else {
            while ( y < e ) {
                *y = activation ( *y, alpha ); // Apply activation function.
                const real t = *z - *y;
                accumulate_average ( ase, t * t, ++cnt );
                ++y; ++z;
            }
        }
        x_ += c++; e += tr_patt;
    }
    const real err = ( ( real { 1 } / real { 2 } ) * ( real ) oput ) * ase;
    std::cout << err << nl;
    return err;
}


template<typename real, typename index, typename sfinae>
typename fccn<real, index, sfinae>::pointer fccn<real, index, sfinae>::nguyen_widrow_weights ( const pointer w_ ) noexcept {
    pointer p = w_;
    static auto dis = std::uniform_real_distribution<real> ( real { -1 }, real { 1 } );
    // Nguyen-Widrow weight initialization.
    for ( const pointer lp = w_ + wgts; p < lp; ++p ) {
        *p = dis ( rng );
    }
    p = w_;
    for ( index nwl = iput + index { 1 }, lnwl = nwl + nrns; nwl < lnwl; ++nwl ) {
        pointer l = p + nwl;
        real norm = index { 0 };
        for ( ; p < l; ++p ) {
            norm += *p * *p;
        }
        p -= nwl;
        const real beta_div_norm = ( real { 7 } / real { 10 } *std::pow ( ( real ) ( nwl ), real { 1 } / ( real ) ( nwl - index { 1 } ) ) ) / std::sqrt ( norm );
        for ( ; p < l; ++p ) {
            *p *= beta_div_norm;
        }
    }
    return w_;
}


template<typename real, typename index, typename sfinae>
typename fccn<real, index, sfinae>::idx_ptr fccn<real, index, sfinae>::construct_jpm_wts_count ( ) const noexcept {
    // Pre-calculate the number of non-zero elements for each (tr_output-)neuron, cummulative in the second half.
    idx_ptr jwc = ( idx_ptr ) scalable_aligned_malloc ( oput * sizeof ( index ), _ALIGN );
    for ( index tr_output_neuron = index { 0 }, i = iput + index { 1 }, o = nrns - oput + tr_output_neuron, acc = index { 0 }; tr_output_neuron < oput; ++tr_output_neuron, ++o ) {
        jwc [ tr_output_neuron ] = ( i + i * o ) + ( o * ( o + index { 1 } ) ) / index { 2 };
    }
    return jwc;
}


template<typename real, typename index, typename sfinae>
typename fccn<real, index, sfinae>::idx_ptr fccn<real, index, sfinae>::construct_nrns_wts_count ( ) const noexcept {
    // Pre-calculate the number of non-zero elements for each (tr_output-)neuron, cummulative in the second half.
    idx_ptr nwc = ( idx_ptr ) scalable_aligned_malloc ( nrns * sizeof ( index ), _ALIGN );
    for ( index tr_output_neuron = index { 0 }, i = iput + index { 1 }, acc = index { 0 }; tr_output_neuron < nrns; ++tr_output_neuron ) {
        nwc [ tr_output_neuron ] = ( acc += i + tr_output_neuron );
    }
    return nwc;
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::fill_wgts_nbn ( ) noexcept {
    // wts needs to have been initialized.
    assert ( nullptr != wts_data );
    pointer t = nbn_data + nrns, f = wts_data + ( index { 2 } *( iput + index { 1 } ) );
    // Copy (relevant) weights to upper.
    for ( index c = index { 1 }; c < nrns; c += index { 1 }, t += nrns, f += iput + c ) { // cc = copy_count
        std::memcpy ( ( void* ) t, ( void* ) f, ( std::size_t ) c * sizeof ( real ) );
    }
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::fill_dlts_nbn ( ) noexcept {
    index incr = nrns + index { 1 };
    pointer t = nullptr, l = nullptr, f = nullptr;
    // Calculate derivatives in the diagonal.
    for ( t = nbn_data, l = t + nrns * nrns, f = pm_data + iput + index { 1 }; t < l; t += incr, ++f ) {
        *t = activation_derivative ( *f, alpha );
    }
    f = nbn_data + nrns;
    // Calculate gains (delta's) in lower.
    for ( incr = index { 1 }; incr < nrns; ++incr, f += nrns ) {
        t = nbn_data;
        for ( index j = index { 0 }; j < incr; ++j, t += nrns ) {
            if constexpr ( std::is_same<value_type, float>::value ) {
                t [ incr ] = f [ incr ] * cblas_sdot ( incr - j, t + j, lapack_int { 1 }, f + j, lapack_int { 1 } );
            }
            else {
                t [ incr ] = f [ incr ] * cblas_ddot ( incr - j, t + j, lapack_int { 1 }, f + j, lapack_int { 1 } );
            }
        }
    }
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::construct_pat ( ) {
    if ( nullptr != pm_data ) {
        scalable_aligned_free ( pm_data );
    }
    if ( nullptr != po_data ) {
        scalable_aligned_free ( po_data );
    }
    pm_data = ( pointer ) scalable_aligned_malloc ( cols * sizeof ( value_type ), _ALIGN );
    pm_front_tr_output = pm_data + cols - oput;
    po_data = ( pointer ) scalable_aligned_malloc ( tr_out->cols * sizeof ( value_type ), _ALIGN );
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::fill_pat ( const index pattern_ ) noexcept {
    // Localize the input (tr_ibo) and the required tr_output (tr_out).
    assert ( tr_patt == tr_out->rows );
    pointer t = nullptr, l = nullptr, f = nullptr;
    for ( t = pm_data, l = t + cols, f = tr_ibo_data + pattern_; t < l; ++t, f += tr_patt ) {
        *t = *f;
    }
    for ( t = po_data, l = t + tr_out->cols, f = tr_out_data + pattern_; t < l; ++t, f += tr_patt ) {
        *t = *f;
    }
    fill_dlts_nbn ( );
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::fill_hes_gra_pm_zero ( const index tr_output_neuron_ ) noexcept {
    // Fill jpm (Jacobian for pattern m)... (size is wgts)
    for ( pointer i = pm_data + iput + index { 1 }, l = pm_front_tr_output + tr_output_neuron_, t = jpm_data, delta_ptr = nbn_data + nrns - oput + tr_output_neuron_; i <= l; ++i, delta_ptr += nrns ) {
        const real delta = -( *delta_ptr ); // deltas.
        for ( pointer p = pm_data; p < i; ++t, ++p ) {
            *t = *p * delta;
        }
    }
    // Fill Hes(sian).
    // Accumulate vector tr_outer product of a line in Jacobian (jpm), elms = number of non-zero elements.
    // Accumulate auto (as in, with itself) vector tr_outer product.
    index elms = jwc_data [ tr_output_neuron_ ], wgts_elms = wgts - elms;
    for ( pointer h = hes_data, jpm_i = jpm_data, jpm_l = jpm_data + elms; jpm_i < jpm_l; h += ++wgts_elms, ++jpm_i ) {
        for ( pointer jpm_j = jpm_i; jpm_j < jpm_l; ++h, ++jpm_j ) {
            *h = *jpm_i * *jpm_j;
        }
    }
    // Fill Gra(dients).
    const real error = po_data [ tr_output_neuron_ ] - pm_front_tr_output [ tr_output_neuron_ ];
    for ( index r = index { 0 }; r < elms; ++r ) {
        gra_data [ r ] = jpm_data [ r ] * error;
    }
    return error * error; // return the squared error.
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::fill_hes_gra_pm ( const index tr_output_neuron_ ) noexcept {
    // Fill jpm (Jacobian for pattern m)... (size is wgts)
    for ( pointer i = pm_data + iput + index { 1 }, l = pm_front_tr_output + tr_output_neuron_, t = jpm_data, delta_ptr = nbn_data + nrns - oput + tr_output_neuron_; i <= l; ++i, delta_ptr += nrns ) {
        const real delta = -( *delta_ptr ); // deltas.
        for ( pointer p = pm_data; p < i; ++t, ++p ) {
            *t = *p * delta;
        }
    }
    // Fill Hes(sian).
    // Accumulate vector tr_outer product of a line in Jacobian (jpm), elms = number of non-zero elements.
    // Accumulate auto (as in, with itself) vector tr_outer product.
    index elms = jwc_data [ tr_output_neuron_ ], wgts_elms = wgts - elms;
    for ( pointer h = hes_data, jpm_i = jpm_data, jpm_l = jpm_data + elms; jpm_i < jpm_l; h += ++wgts_elms, ++jpm_i ) {
        for ( pointer jpm_j = jpm_i; jpm_j < jpm_l; ++h, ++jpm_j ) {
            *h += *jpm_i * *jpm_j;
        }
    }
    // Fill Gra(dients).
    const real error = po_data [ tr_output_neuron_ ] - pm_front_tr_output [ tr_output_neuron_ ];
    for ( index r = index { 0 }; r < elms; ++r ) {
        gra_data [ r ] += jpm_data [ r ] * error;
    }
    return error * error; // return the squared error.
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::fill_dia_hes_gra_ase ( ) {
    fill_wgts_nbn ( );
    fill_pat ( index { 0 } );
    const index l = oput - index { 1 };
    index cnt = index { 2 };
    real ase = fill_hes_gra_pm_zero ( l ); // We have to do the last neuron of this (first) pattern first, so the whole hes/gra vectors get "reset".
    for ( index tr_output_neuron = index { 0 }; tr_output_neuron < l; ++tr_output_neuron, ++cnt ) {
        accumulate_average ( ase, fill_hes_gra_pm ( tr_output_neuron ), cnt );
    }
    for ( index pattern = index { 1 }; pattern < tr_patt; ++pattern ) {
        fill_pat ( pattern );
        // For every tr_output neuron, only the non-zero elements.
        for ( index tr_output_neuron = index { 0 }; tr_output_neuron < oput; ++tr_output_neuron, ++cnt ) {
            accumulate_average ( ase, fill_hes_gra_pm ( tr_output_neuron ), cnt );
        }
    }
    // Copy lower (Hessian) to upper triangular and diagonal to dia.
    {
        const index incr_t = wgts + index { 1 };
        index incr_f = index { 0 };
        const pointer l = hes_data + wgts_sqr;
        pointer d = dia_data, f = hes_data, h;
        for ( pointer t = hes_data + wgts; t < l; t += incr_t, f += ++incr_f ) {
            *d++ = *f++; // Copy diagonal... (except last element, see below)
            for ( h = t; h < l; h += wgts, ++f ) {
                *h = *f;
            }
        }
        *d = *f; // Last element tot be copied.
    }
    return ( ( real { 1 } / real { 2 } ) * ( real ) oput ) * ase;
}


#include "fccn_invert.inl"


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::update_wts ( ) noexcept {
    if constexpr ( std::is_same<value_type, float>::value ) {
        cblas_sgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) wgts, ( lapack_int ) wgts, 1.0f, ihs_data, ( lapack_int ) wgts, gra_data, 1, 0.0f, dlt_data, 1 );
    }
    else {
        cblas_dgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) wgts, ( lapack_int ) wgts, 1.0, ihs_data, ( lapack_int ) wgts, gra_data, 1, 0.0, dlt_data, 1 );
    }
    pointer t = wts_pdata, f = wts_data;
    for ( pointer d = dlt_data, l = d + wgts; d < l; ++t, ++f, ++d ) {
        *t = *f - *d;
    }
    t = dlt_data; f = wts_data; dlt_data = dlt_pdata; wts_data = wts_pdata; dlt_pdata = t; wts_pdata = f;
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::update_wts_asw ( ) noexcept { // And calculate the average squared weights.
    if constexpr ( std::is_same<value_type, float>::value ) {
        cblas_sgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) wgts, ( lapack_int ) wgts, 1.0f, ihs_data, ( lapack_int ) wgts, gra_data, 1, 0.0f, dlt_data, 1 );
    }
    else {
        cblas_dgemv ( CblasColMajor, CblasNoTrans, ( lapack_int ) wgts, ( lapack_int ) wgts, 1.0, ihs_data, ( lapack_int ) wgts, gra_data, 1, 0.0, dlt_data, 1 );
    }
    real asw = real { 0 };
    index cnt = index { 1 };
    pointer t = wts_pdata, f = wts_data;
    for ( pointer d = dlt_data, l = d + wgts; d < l; ++t, ++f, ++d, ++cnt ) {
        *t = *f - *d;
        accumulate_average ( asw, *t * *t, cnt );
    }
    asw *= real { 1 } / real { 2 };
    t = dlt_data; f = wts_data; dlt_data = dlt_pdata; wts_data = wts_pdata; dlt_pdata = t; wts_pdata = f;
    return asw;
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::reset_wts ( ) noexcept {
    static pointer t = nullptr, f = nullptr;
    t = dlt_data; f = wts_data; dlt_data = dlt_pdata; wts_data = wts_pdata; dlt_pdata = t; wts_pdata = f;
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::update_hes ( const real comb_coeff_ ) noexcept {
    const index incr = wgts + 1;
    for ( pointer t = hes_data, f = dia_data, l = f + wgts; f < l; t += incr, ++f ) {
        *t = *f + comb_coeff_;
    }
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::levenberg_marquardt_train1 ( index epochs_ ) {
    constexpr real comb_coeff_divider = real { 1 } / real { 9 }, comb_coeff_multiplier = real { 10 }, comb_coeff_min = std::numeric_limits<real>::epsilon ( ) * comb_coeff_multiplier, comb_coeff_max = std::numeric_limits<real>::max ( ) / comb_coeff_multiplier;
    feedforward ( );
    real curr_ase, prev_ase = fill_dia_hes_gra_ase ( ), comb_coeff = real { 1 } / real { 10 };
    real max_ase = ( real ) 0.00075f;
    index c = 0;
    index m = index { 0 };
    while ( true ) {
        while ( levenberg_marquardt_cholesky_invert_hes ( ) ) {
            // The Hessian is singular, continue to the next iteration until
            // the diagonal update transforms it back to non-singular.
            update_hes ( comb_coeff *= comb_coeff_multiplier );
        }
        update_wts ( );
        curr_ase = feedforward_ase ( wts_data );
        std::printf ( "%3u - sse: %.8f\n", c++, curr_ase );
        if ( curr_ase <= max_ase or not ( epochs_ ) ) {
            std::printf ( "sse: %.8f\n", curr_ase );
            break;
        }
        else if ( curr_ase <= prev_ase ) {
            comb_coeff *= comb_coeff_divider;
        }

        else if ( m++ < 5 ) {
            reset_wts ( );
            update_hes ( comb_coeff *= comb_coeff_multiplier );
            continue;
        }
        m = index { 0 };
        prev_ase = fill_dia_hes_gra_ase ( );
        update_hes ( comb_coeff );
        --epochs_;
    }
    return curr_ase;
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::levenberg_marquardt_train2 ( ) {
    // http://crsouza.com/2009/11/18/neural-network-learning-by-the-levenberg-marquardt-algorithm-with-bayesian-regularization-part-2/
    constexpr real comb_coeff_divider = real { 1 } / real { 10 }, comb_coeff_multiplier = real { 10 }, comb_coeff_min = std::numeric_limits<real>::epsilon ( ) * comb_coeff_multiplier, comb_coeff_max = std::numeric_limits<real>::max ( ) / comb_coeff_multiplier;
    feedforward ( );
    real curr_ase, prev_ase = fill_dia_hes_gra_ase ( ), asw = average_squared_weights ( ), comb_coeff = real { 1 } / real { 10 };
    real max_ase = ( real ) 0.0001f, bayes_alpha = real { 0.0f }, bayes_beta = real { 1.0f };
    const index incr = wgts + 1;
    index c = 0;
    index m = index { 0 };
    real prev_bayes_cost = bayes_beta * prev_ase + bayes_alpha * asw, curr_bayes_cost;
    while ( true ) {
        for ( pointer t = hes_data, f = dia_data, l = f + wgts; f < l; t += incr, ++f ) {
            *t = *f + comb_coeff;
        }
        const lapack_int info = levenberg_marquardt_aasen_invert_hes ( );
        if ( lapack_int { 0 } != info ) {
            // The Hessian is singular, continue to the next iteration until
            // the diagonal update transforms it back to non-singular.
            comb_coeff *= comb_coeff_multiplier;
            continue;
        }
        asw = update_wts_asw ( );
        curr_ase = feedforward_ase ( );
        curr_bayes_cost = bayes_beta * curr_ase + bayes_alpha * asw;
        std::printf ( "%3u - bayes cost: %.8f\n", c++, curr_bayes_cost );
        if ( curr_ase <= max_ase ) {
            break;
        }
        else if ( curr_bayes_cost <= prev_bayes_cost ) {
            comb_coeff *= comb_coeff_divider;
        }
        else if ( m < 5 ) {
            comb_coeff *= comb_coeff_multiplier;
            ++m;
            reset_wts ( );
            continue;
        }
        m = index { 0 };
        prev_ase = fill_dia_hes_gra_ase ( );
        /*

        Update the Bayesian hyperparameters using MacKay�s or Poland�s formulae:

        gamma = W � (alpha * tr(H-1))
        beta = (N � gamma) / 2.0 * Ed
        alpha = W / (2.0 * Ew + tr(H-1)) [modified Poland�s update], or
        alpha = gamma / (2.0 * Ew) [original MacKay�s update], where:
        W is the number of network parameters (number of weights and biases)
        N is the number of entries in the training set
        tr(H-1) is the trace of the inverse Hessian matrix

        */
        const real ihs_trace = ihs->trace ( ), gamma = wgts - bayes_alpha * ihs_trace;
        bayes_beta = ( tr_patt - gamma ) / ( real { 2 } * prev_ase );
        // Poland's update formula.
        bayes_alpha = wgts / ( real { 2 } * asw + ihs_trace );
        // MacKay's update formula.
        // bayes_alpha = gamma / ( real { 2 } * asw );
        prev_bayes_cost = bayes_beta * prev_ase + bayes_alpha * asw;
    }

    return curr_ase;
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::accumulate_average ( real & avg_, const real value_, const index cnt_ ) noexcept {
    // Calculates the average on the fly.
    avg_ += ( value_ - avg_ ) / ( real ) cnt_;
}


template<typename real, typename index, typename sfinae>
std::vector<index> fccn<real, index, sfinae>::to_gray ( index value_, index extended_digits_ ) noexcept {
    constexpr index base = index { 3 };
    // https://en.wikipedia.org/wiki/Gray_code
    const index digits = sax::iLog<base> ( value_ ) + index { 1 };
    static std::vector<index> baseN; // Stores the ordinary base-N number, one digit per entry.
    baseN.resize ( digits );
    index i; // The loop variable.
    // Put the normal baseN number into the baseN array. For base 10, 109
    // would be stored as [9,0,1].
    for ( i = index { 0 }; i < digits; ++i ) {
        baseN [ i ] = value_ % base;
        value_ /= base;
    }
    // Convert the normal baseN number into the Gray code equivalent. Note that
    // the loop starts at the most significant digit and goes down.
    value_ = index { 0 }; // value_ is (has become) the shift.
    std::vector<index> gray ( extended_digits_ );
    while ( i-- ) {
        // The gray digit gets shifted down by the sum of the higher digits.
        gray [ i ] = ( baseN [ i ] + value_ ) % base;
        value_ += base - gray [ i ]; // Subtract from base so shift is positive.
    }
    return gray;
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::average_squared_errors ( ) const noexcept {
    // Average of Squared Errors.
    real ase = real { 0 };
    pointer od = tr_out_data, id = tr_ibo_data + ( iput + index { 1 } +nrns - oput ) * tr_patt;
    for ( index cnt = index { 1 }, l = oput * tr_patt + index { 1 }; cnt < l; ++cnt, ++od, ++id ) {
        const real error = *od - *id;
        accumulate_average ( ase, error * error, cnt );
    }
    return ( ( real { 1 } / real { 2 } ) * ( real ) oput ) * ase;
}


template<typename real, typename index, typename sfinae>
real fccn<real, index, sfinae>::average_squared_weights ( ) const noexcept {
    // Average of Squared Weights.
    real asw = real { 0 };
    pointer w = wts_data;
    for ( index cnt = index { 1 }; cnt <= wgts; ++cnt, ++w ) {
        accumulate_average ( asw, *w * *w, cnt );
    }
    return ( real { 1 } / real { 2 } ) * asw;
}


template<typename real, typename index, typename sfinae>
typename fccn<real, index, sfinae>::pointer fccn<real, index, sfinae>::convert ( const fmat *m_ ) {
    const index l = m_->n_rows * m_->n_cols;
    pointer t = ( pointer ) scalable_aligned_malloc ( l * sizeof ( value_type ), _ALIGN );
    auto f = m_->v;
    for ( index i = index { 0 }; i < l; ++i ) {
        t [ i ] = ( value_type ) f [ i ];
    }
    return t;
}


template<typename real, typename index, typename sfinae>
void fccn<real, index, sfinae>::copy_wts ( const fmat *m_ ) {
    const index l = m_->n_rows * m_->n_cols;
    auto f = m_->v;
    for ( index i = index { 0 }; i < l; ++i ) {
        wts_data [ i ] = ( value_type ) f [ i ];
    }
}


template<typename real, typename index, typename sfinae>
template<class Archive>
void fccn<real, index, sfinae>::save ( Archive & ar_ ) const {
    ar_ ( nrns, iput, oput, wgts, wgts_sqr, cols, alpha );
    ar_ ( cereal::binary_data ( wts_data, wgts * ( index ) sizeof ( real ) ) );
}


template<typename real, typename index, typename sfinae>
template<class Archive>
void fccn<real, index, sfinae>::load ( Archive & ar_ ) {
    ar_ ( nrns, iput, oput, wgts, wgts_sqr, cols, alpha );
    ar_ ( cereal::binary_data ( wts_data, wgts * ( index ) sizeof ( real ) ) );
}


template<typename real, typename index, typename sfinae>
auto fccn<real, index, sfinae>::seed_rng ( ) noexcept {
    return sax::os_seed ( );
}
