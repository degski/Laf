
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
matrix<real, index, sfinae>::matrix ( pointer ptr_, const index rows_, const index cols_, matrix_creation ct_, const real lb_, const real ub_ ) :

    m_data ( ptr_ ), rows ( rows_ ), cols ( cols_ ) {

    switch ( ct_ ) {

        case matrix_creation::none: break;
        case matrix_creation::zero: zero ( this ); break;
        case matrix_creation::fill: fill ( this, lb_ ); break;
        case matrix_creation::gaussian: gaussian ( this ); break;
        case matrix_creation::uniform: uniform ( this, lb_, ub_ ); break;
    }

    if ( rows == cols ) {

        m_data = tag_is_square ( m_data );
    }
}


template<typename real, typename index, typename sfinae>
matrix<real, index, sfinae>::matrix ( const index rows_, const index cols_, matrix_creation ct_, const real lb_, const real ub_ ) :

    matrix ( tag_is_not_assigned ( ( pointer ) scalable_aligned_malloc ( rows_ * cols_ * ( index ) sizeof ( value_type ), alignment ( ) ) ), rows_, cols_, ct_, lb_, ub_ ) { }


template<typename real, typename index, typename sfinae>
matrix<real, index, sfinae>::matrix ( const std::pair<index, index> & dims_, matrix_creation ct_, const real lb_, const real ub_ ) :

    matrix ( tag_is_not_assigned ( ( pointer ) scalable_aligned_malloc ( dims_.first * dims_.second * ( index ) sizeof ( value_type ), alignment ( ) ) ), dims_.first, dims_.second, ct_, lb_, ub_ ) { }


template<typename real, typename index, typename sfinae>
matrix<real, index, sfinae>::~matrix ( ) {

    if ( is_not_assigned ( m_data ) ) {

        scalable_aligned_free ( untag ( m_data ) );
    }
}


template<typename real, typename index, typename sfinae>
std::pair<typename matrix<real, index, sfinae>::matrix_ptr, lapack_int> matrix<real, index, sfinae>::invert ( ) const {

    assert ( is_square ( ) );

    //	Calculates the inverse of the n * n matrix X: Y = X^-1.
    //	Does not change the value of X, unless Y = X...

    std::pair<matrix_ptr, lapack_int> rv { construct_copy ( ), 0 };

    lapack_int *lapack_ipiv = ( lapack_int* ) scalable_aligned_malloc ( rv.first->rows * sizeof ( lapack_int ), alignment ( ) );

    if constexpr ( std::is_same<value_type, float>::value ) {

        rv.second = LAPACKE_sgetrf ( CblasColMajor, ( lapack_int ) rv.first->rows, ( lapack_int ) rv.first->cols, rv.first->data ( ), ( lapack_int ) rv.first->rows, lapack_ipiv );

        if ( not ( rv.second ) ) {

            rv.second = LAPACKE_sgetri ( CblasColMajor, ( lapack_int ) rv.first->rows, rv.first->data ( ), ( lapack_int ) rv.first->rows, lapack_ipiv );
        }
    }

    else {

        rv.second = LAPACKE_dgetrf ( CblasColMajor, ( lapack_int ) rv.first->rows, ( lapack_int ) rv.first->cols, rv.first->data ( ), ( lapack_int ) rv.first->rows, lapack_ipiv );

        if ( not ( rv.second ) ) {

            rv.second = LAPACKE_dgetri ( CblasColMajor, ( lapack_int ) rv.first->rows, rv.first->data ( ), ( lapack_int ) rv.first->rows, lapack_ipiv );
        }
    }

    scalable_aligned_free ( lapack_ipiv );

    return rv;
}


template<typename real, typename index, typename sfinae>
std::pair<typename matrix<real, index, sfinae>::matrix_ptr, lapack_int> matrix<real, index, sfinae>::solve ( const matrix_ptr b_ ) const {

    assert ( is_square ( ) );
    assert ( rows == b_->rows );

    // Solves the matrix equation A*x = B for x. A is an n * n matrix in column major vector notation.
    // B and x are vectors of lenght n. A and B are not changed. x will contain the solution.
    // x can equal B if B may be overwritten...

    std::pair<matrix_ptr, lapack_int> rv { b_->construct_copy ( ), 0 };

    lapack_int *lapack_ipiv = ( lapack_int* ) scalable_aligned_malloc ( rows * sizeof ( lapack_int ), alignment ( ) );

    if constexpr ( std::is_same<value_type, float>::value ) {

        rv.second = LAPACKE_sgesv ( CblasColMajor, ( lapack_int ) rows, ( lapack_int ) b_->cols, data ( ), ( lapack_int ) rows, lapack_ipiv, rv.first->data ( ), ( lapack_int ) rv.first->rows );
    }

    else {

        rv.second = LAPACKE_dgesv ( CblasColMajor, ( lapack_int ) rows, ( lapack_int ) b_->cols, data ( ), ( lapack_int ) rows, lapack_ipiv, rv.first->data ( ), ( lapack_int ) rv.first->rows );
    }

    scalable_aligned_free ( lapack_ipiv );

    return rv;
}


template<typename real, typename index, typename sfinae>
real matrix<real, index, sfinae>::trace ( ) const noexcept {

    assert ( is_square ( ) );

    real t = real { 0 };

    const index rows_1 = rows + 1;

    for ( pointer d = data ( ), l = d + rows * cols; d < l; d += rows_1 ) {

        t += *d;
    }

    return t;
}


template<typename real, typename index, typename sfinae>
std::pair<typename matrix<real, index, sfinae>::value_type, typename matrix<real, index, sfinae>::value_type> matrix<real, index, sfinae>::min_max ( ) const noexcept {

    // Simultaneous min & max using only 3 * N / 2 comparisons...

    if ( rows == 0 || cols == 0 ) {

        return { NAN, NAN };
    }

    const pointer d = data ( );

    std::pair<value_type, value_type> min_max;

    if ( rows * cols == 1 ) {

        return { d [ 0 ], d [ 0 ] };
    }

    else {

        if ( d [ 0 ] > d [ 1 ] ) {

            min_max.second = d [ 0 ]; min_max.first = d [ 1 ];
        }

        else {

            min_max.first = d [ 0 ]; min_max.second = d [ 1 ];
        }
    }

    const index last = rows * cols;

    for ( index i0 = 0, i1 = 1; i1 < last; ++i0, ++i1 ) {

        if ( d [ i0 ] > d [ i1 ] ) {

            if ( min_max.second < d [ i0 ] ) min_max.second = d [ i0 ];
            if ( min_max.first > d [ i1 ] ) min_max.first = d [ i1 ];
        }

        else {

            if ( min_max.second < d [ i1 ] ) min_max.second = d [ i1 ];
            if ( min_max.first > d [ i0 ] ) min_max.first = d [ i0 ];
        }
    }

    return min_max;
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::reference matrix<real, index, sfinae>::at ( const index r_, const index c_ ) noexcept {

    return data ( ) [ c_ * rows + r_ ];
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::value_type matrix<real, index, sfinae>::at ( const index r_, const index c_ ) const noexcept {

    return data ( ) [ c_ * rows + r_ ];
}


template<typename real, typename index, typename sfinae>
void matrix<real, index, sfinae>::transpose ( ) {

    if constexpr ( std::is_same<value_type, float>::value ) {

        mkl_simatcopy ( 'C', 'T', rows, cols, real { 1 }, data ( ), rows, cols );
    }

    else {

        mkl_dimatcopy ( 'C', 'T', rows, cols, real { 1 }, data ( ), rows, cols );
    }

    std::swap ( rows, cols );
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::construct_copy ( ) const {

    matrix_ptr const m = ( matrix_ptr ) scalable_aligned_malloc ( matrix_size ( ) + data_size ( ), alignment ( ) );

    std::memcpy ( ( void* ) m, ( void* ) this, matrix_size ( ) + data_size ( ) );
    m->m_data = rows == cols ? tag_is_square ( ( pointer ) ( ( ( char* ) m ) + matrix_size ( ) ) ) : ( pointer ) ( ( ( char* ) m ) + matrix_size ( ) );

    return m;
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::construct_transpose ( ) const {

    matrix_ptr const c = construct ( cols, rows );

    if constexpr ( std::is_same<value_type, float>::value ) {

        mkl_somatcopy ( 'C', 'T', rows, cols, real { 1 }, data ( ), rows, c->data ( ), c->rows );
    }

    else {

        mkl_domatcopy ( 'C', 'T', rows, cols, real { 1 }, data ( ), rows, c->data ( ), c->rows );
    }

    return c;
}


template<typename real, typename index, typename sfinae>
void matrix<real, index, sfinae>::fill ( const real v_ ) {

    for ( pointer d = data ( ), l = d + rows * cols; d < l; ++d ) {

        *d = v_;
    }
}


template<typename real, typename index>
index real_as_index ( const real v_ ) noexcept {

    if constexpr ( sizeof ( index ) <= sizeof ( real ) ) {

        index r;

        std::memcpy ( &r, &v_, sizeof ( real ) );

        return r;
    }

    else { // real = double and index = std::uint32_t...

        std::uintptr_t r;

        std::memcpy ( &r, &v_, sizeof ( real ) );

        return ( index ) ( r ^ ( r >> 32 ) );
    }
}


template<typename real, typename index, typename sfinae>
index matrix<real, index, sfinae>::hash ( ) const noexcept {

    index h = sax::hash ( rows ) ^ sax::hash ( sax::hash ( cols ) * hash_multiplier ( ) );

    const index last = rows * cols;
    const pointer d = data ( );

    for ( index i = 0; i < last; ++i ) {

        h ^= sax::hash ( real_as_index<real, index> ( d [ i ] ) );
    }

    return h;
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::construct ( const index rows_, const index cols_ ) {

    matrix_ptr const m = ( matrix_ptr ) scalable_aligned_malloc ( matrix_size ( ) + rows_ * cols_ * ( index ) sizeof ( real ), alignment ( ) );

    m->m_data = rows_ == cols_ ? tag_is_square ( ( pointer ) ( ( ( char* ) m ) + matrix_size ( ) ) ) : ( pointer ) ( ( ( char* ) m ) + matrix_size ( ) );
    m->rows = rows_;
    m->cols = cols_;

    return m;
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::zero ( const typename matrix<real, index, sfinae>::matrix_ptr m_ ) {

    std::memset ( ( void* ) m_->data ( ), 0, ( std::size_t ) m_->data_size ( ) );

    return m_;
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::gaussian ( const typename matrix<real, index, sfinae>::matrix_ptr m_ ) {

    // gaussian with mean 1/4 or -1/4 and standard deviation 1/4...

    std::bernoulli_distribution bdis;
    std::normal_distribution<real> ndis ( ( real { 1 } / real { 4 } ), ( real { 1 } / real { 8 } ) );

    const index last = m_->rows * m_->cols;
    const pointer d = m_->data ( );

    for ( index i = 0; i < last; ++i ) {

        d [ i ] = bdis ( rng ) ? ndis ( rng ) : -ndis ( rng );
    }

    return m_;
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::uniform ( const typename matrix<real, index, sfinae>::matrix_ptr m_, const real lb_, const real ub_ ) {

    std::uniform_real_distribution<real> dis ( lb_, ub_ );

    const index last = m_->rows * m_->cols;
    const pointer d = m_->data ( );

    for ( index i = 0; i < last; ++i ) {

        while ( std::abs ( d [ i ] = dis ( rng ) ) < ( real { 1 } / real { 10 } ) );
    }

    return m_;
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::fill ( const typename matrix<real, index, sfinae>::matrix_ptr m_, const real v_ ) {

    const index last = m_->rows * m_->cols;
    const pointer d = m_->data ( );

    for ( index i = 0; i < last; ++i ) {

       d [ i ] = v_;
    }

    return m_;
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::construct_zero ( const index rows_, const index cols_ ) {

    return zero ( construct ( rows_, cols_ ) );
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::construct_fill ( const index rows_, const index cols_, const real v_ ) {

    return fill ( construct ( rows_, cols_ ), v_ );
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::construct_gaussian ( const index rows_, const index cols_ ) {

    return gaussian ( construct ( rows_, cols_ ) );
}


template<typename real, typename index, typename sfinae>
typename matrix<real, index, sfinae>::matrix_ptr matrix<real, index, sfinae>::construct_uniform ( const index rows_, const index cols_, const real lb_, const real ub_ ) {

    return uniform ( construct ( rows_, cols_ ), lb_, ub_ );
}


template<typename real, typename index, typename sfinae>
void matrix<real, index, sfinae>::destroy ( typename matrix<real, index, sfinae>::matrix_ptr & ptr_ ) noexcept {

    if ( nullptr != ptr_ ) {

        ptr_->~matrix ( );
        scalable_aligned_free ( ptr_ );
    }
}


template<typename real, typename index, typename sfinae>
template<class Archive>
void matrix<real, index, sfinae>::save ( Archive & ar_ ) const {

    ar_ ( rows, cols );
    ar_ ( cereal::binary_data ( data ( ), rows * cols * ( index ) sizeof ( real ) ) );
}


template<typename real, typename index, typename sfinae>
template<class Archive>
void matrix<real, index, sfinae>::load ( Archive & ar_ ) {

    ar_ ( rows, cols );

    if ( is_not_assigned ( m_data ) ) {

        scalable_aligned_free ( untag ( m_data ) );
        m_data = tag_is_not_assigned ( ( pointer ) scalable_aligned_malloc ( rows * cols * ( index ) sizeof ( real ), alignment ( ) ) );

        if ( rows == cols ) {

            m_data = tag_is_square ( m_data );
        }
    }

    else if ( nullptr == m_data ) {

        m_data = tag_is_not_assigned ( ( pointer ) scalable_aligned_malloc ( rows * cols * ( index ) sizeof ( real ), alignment ( ) ) );

        if ( rows == cols ) {

            m_data = tag_is_square ( m_data );
        }
    }

    ar_ ( cereal::binary_data ( data ( ), rows * cols * ( index ) sizeof ( real ) ) );
}


template<typename real, typename index, typename sfinae>
auto matrix<real, index, sfinae>::seed_rng ( ) noexcept {
    return sax::os_seed ( );
}
