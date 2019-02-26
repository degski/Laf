
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


namespace detail {

template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
inline index float_width ( const real r_, const index precision_ ) noexcept {

    if ( r_ == real { 0 } ) {

        return index { 4 };
    }

    //                          leading digits          |  minus sign  | dot + trailing space | decimal digits | -1 and 1 have a leading 0 (and 0 leading digits)
 
    return ( ( index ) std::ceil ( std::log10 ( std::abs ( r_ ) ) ) + ( r_ < real { 0 } ) + index { 2 } + precision_ + ( r_ >= real { -1 } && r_ <= real { 1 } ) );
}



template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
inline index format_width ( const matrix<real, index, sfinae> & m_, const index precision_ ) noexcept {

    const std::pair<real, real> min_max = m_.min_max ( );

    return ( index ) std::max ( float_width ( min_max.first, precision_ ), float_width ( min_max.second, precision_ ) );
}


template<typename index>
inline char *make_float_format_string ( const index precision_, const index width_ ) {

    char *s = ( char* ) std::malloc ( 8 * sizeof ( char ) ), t [ 8 ];

    std::strcpy ( s, "%" ); _itoa ( width_, t, 10 ); std::strcat ( s, t ); std::strcat ( s, "." ); _itoa ( precision_, t, 10 ); std::strcat ( s, t ); std::strcat ( s, "f" );

    return s;
}


template<typename index>
inline char *make_int_format_string ( const index width_ ) {

    char *s = ( char* ) std::malloc ( 8 * sizeof ( char ) ), t [ 8 ];

    std::strcpy ( s, "%" ); _itoa ( width_, t, 10 ); std::strcat ( s, t ); std::strcat ( s, "u" );

    return s;
}


template<typename index>
inline char *make_zero_string ( const index width_ ) {

    char *s = ( char* ) std::malloc ( 32 * sizeof ( char ) );

    std::strcpy ( s, " " );

    for ( index i = 2; i < width_; i++ ) {

        std::strcat ( s, " " );
    }

    std::strcat ( s, "-" );

    return ( char* ) std::realloc ( s, std::strlen ( s ) + 1 );
}


template<typename index>
inline char *make_empty_string ( const index width_ ) {

    char *s = ( char* ) std::malloc ( 32 * sizeof ( char ) );

    std::strcpy ( s, " " );

    for ( index i = 1; i < width_; i++ ) {

        std::strcat ( s, " " );
    }

    return ( char* ) std::realloc ( s, std::strlen ( s ) + 1 );
}


template<typename index>
inline constexpr index power_of_10 ( const index p_ ) noexcept {

    constexpr index table [ 10 ] = { 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000 };

    return table [ std::min ( p_, index { 9 } ) ];
}


template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
inline void print_characteristics ( const matrix<real, index, sfinae> & m_, const char *s_ ) noexcept {

    if constexpr ( sizeof ( index ) == 4 ) {

        std::printf ( "%srows %lu, cols %lu, elem %lu\n%shash %lu\n\n", s_, ( unsigned long ) m_.rows, ( unsigned long ) m_.cols, ( unsigned long ) ( m_.rows * m_.cols ), s_, ( unsigned long ) m_.hash ( ) );
    }

    else {

        std::printf ( "%srows %llu, cols %llu, elem %llu\n%shash %llu\n\n", s_, ( unsigned long long ) m_.rows, ( unsigned long long ) m_.cols, ( unsigned long long ) ( m_.rows * m_.cols ), s_, ( unsigned long long ) m_.hash ( ) );
    }
}


template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
void print ( const matrix<real, index, sfinae> & m_, const index precision_ ) {

    using real_ptr = real*;

    if ( m_.rows == 0 or m_.cols == 0 ) {

        std::printf ( "empty matrix\n" );

        return;
    }

    const index width = format_width ( m_, precision_ ), scale = power_of_10 ( width - index { 2 } );

    char *l_fs = make_float_format_string ( precision_, width ), *l_us = make_int_format_string ( width ), *l_zs = make_zero_string ( width ), *l_es = make_empty_string ( width );

    const index rows = m_.rows, elem = m_.rows * m_.cols;
    const real_ptr next = m_.data ( ) + rows;
    const real precisionf = ( real ) power_of_10 ( precision_ );

    print_characteristics ( m_, l_es );

    std::printf ( "%s", l_es );

    index i;

    for ( i = 0; i < m_.cols; ++i ) {

        std::printf ( l_us, i % scale );
    }

    std::printf ( "\n\n" );

    i = 0;

    for ( real_ptr d = m_.data ( ); d < next; ++d, ++i ) {

        std::printf ( l_us, i % scale );

        for ( index j = 0; j < elem; j += rows )

            if ( ( index ) ( std::fabs ( *( d + j ) ) * precisionf + ( real { 1 } / real { 2 } ) ) == 0 ) {

                std::printf ( "%s", l_zs );
            }

            else {

                std::printf ( l_fs, *( d + j ) ); // print ColMajor!
            }

            std::printf ( "\n" );
    }

    free ( l_fs ); free ( l_us ); free ( l_zs ); free ( l_es );
}

}
