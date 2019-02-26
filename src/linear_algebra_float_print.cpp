
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

#include "linear_algebra_float.h"


// Internal printing stuff...

inline std::uint32_t linear_algebra_float_float_width ( const float f, const std::uint32_t precision ) {

    if ( f == 0.0f ) {

        return std::uint32_t { 4 };
    }

	//                          leading digits          |  minus sign  | dot + trailing space | decimal digits | -1 and 1 have a leading 0 (and 0 leading digits)

	return ( ( std::uint32_t ) ceilf ( log10f ( fabsf ( f ) ) ) + ( f < 0.0f ) +          2           +   precision    +          ( f >= -1.0f && f <= 1.0f ) );
}

inline std::uint32_t linear_algebra_float_format_width ( const fmat *m, const std::uint32_t precision ) {

	float _min, _max;

	linear_algebra_float_min_max ( &_min, &_max, m );

	return MAX ( linear_algebra_float_float_width ( _min, precision ), linear_algebra_float_float_width ( _max, precision ) );
}

inline char *linear_algebra_float_make_float_format_string ( const std::uint32_t precision, const std::uint32_t width ) {

	char *s = ( char* ) malloc ( 8 * sizeof ( char ) ), t [ 8 ];

	strcpy ( s, "%" ); _itoa ( width, t, 10 ); strcat ( s, t ); strcat ( s, "." ); _itoa ( precision, t, 10 ); strcat ( s, t ); strcat ( s, "f" );

	return s;
}

inline char *linear_algebra_float_make_int_format_string ( const std::uint32_t width ) {

	char *s = ( char* ) malloc ( 8 * sizeof ( char ) ), t [ 8 ];

	strcpy ( s, "%" ); _itoa ( width, t, 10 ); strcat ( s, t ); strcat ( s, "u" );

	return s;
}

inline char *linear_algebra_float_make_zero_string ( const std::uint32_t width ) {

	char *s = ( char* ) malloc ( 32 * sizeof ( char ) );

	strcpy ( s, " " );

	for ( std::uint32_t i = 2; i < width; i++ ) {

		strcat ( s, " " );
	}

	strcat ( s, "-" );

	return ( char* ) realloc ( s, strlen ( s ) +1 );
}

inline char *linear_algebra_float_make_empty_string ( const std::uint32_t width ) {

	char *s = ( char* ) malloc ( 32 * sizeof ( char ) );

	strcpy ( s, " " );

	for ( std::uint32_t i = 1; i < width; i++ ) {

		strcat ( s, " " );
	}

	return ( char* ) realloc ( s, strlen ( s ) +1 );
}


inline std::uint32_t uword_power_of_10 ( const std::uint32_t p ) {

	const std::uint32_t table [ 10 ] = { 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000 };

	return table [ MIN ( p, 9 ) ];
}


inline void linear_algebra_float_print_characteristics ( const fmat *m, const char *s ) {

	printf ( "c-impl: %srows %u, cols %u, elem %u\n%shash %u\n\n", s, m->n_rows, m->n_cols, m->n_elem, s, linear_algebra_float_hash ( m ) );
}


void linear_algebra_float_print ( const fmat *m, const std::uint32_t precision ) {

	if ( m == nullptr ) {

		printf ( "undefined matrix\n" );

		return;
	}

	if ( m->n_elem == 0 ) {

		printf ( "empty matrix\n" );

		return;
	}

	const std::uint32_t width = linear_algebra_float_format_width ( m, precision ), scale = uword_power_of_10 ( width -2 );

	char *_fs = linear_algebra_float_make_float_format_string ( precision, width ), *_us = linear_algebra_float_make_int_format_string ( width ), *_zs = linear_algebra_float_make_zero_string ( width ), *_es = linear_algebra_float_make_empty_string ( width );

	const std::uint32_t n_rows = m->n_rows, n_elem = m->n_elem;
	const float *next = m->v + n_rows, precisionf = ( float ) uword_power_of_10 ( precision );

	linear_algebra_float_print_characteristics ( m, _es );

	printf ( "%s", _es );

	std::uint32_t i;

	for ( i = 0; i < m->n_cols; i++ ) {

		printf ( _us, i % scale );
	}

	printf ( "\n\n" );

	i = 0;

	for ( float *v = m->v; v < next; v++, i++ ) {

		printf ( _us, i % scale );

		for ( std::uint32_t j = 0; j < n_elem; j += n_rows )

			if ( ( std::uint32_t ) ( fabsf ( *( v + j ) ) * precisionf + 0.5f ) == 0 ) {

				printf ( "%s", _zs );
			}

			else {

				printf ( _fs, *( v + j ) ); // print ColMajor!
			}

		printf ( "\n" );
	}

	printf ( "\n" );

	free ( _fs ); free ( _us ); free ( _zs ); free ( _es );
}


void linear_algebra_float_print_transposed ( fmat *m, const std::uint32_t precision ) {

    std::swap ( m->n_rows, m->n_cols );
    linear_algebra_float_print ( m, precision );
    std::swap ( m->n_rows, m->n_cols );
}


void linear_algebra_float_print_rm ( const fmat *m, const std::uint32_t precision ) {

	if (         m == nullptr ) { printf ( "undefined matrix\n" ); return; }
	if ( m->n_elem == 0    ) { printf ( "empty matrix\n" ); return; }

	const std::uint32_t width = linear_algebra_float_format_width ( m, precision ), scale = uword_power_of_10 ( width -2 );

	char *_fs = linear_algebra_float_make_float_format_string ( precision, width ), *_us = linear_algebra_float_make_int_format_string ( width ), *_zs = linear_algebra_float_make_zero_string ( width ), *_es = linear_algebra_float_make_empty_string ( width );

	const float precisionf = ( float ) uword_power_of_10 ( precision );

	linear_algebra_float_print_characteristics ( m, _es );

	printf ( "%s", _es );

	std::uint32_t i;

	for ( i = 0; i < m->n_cols; i++ ) {

		printf ( _us, i % scale );
	}

	printf ( "\n\n" );

	i = 0;

	for ( i = 0; i < m->n_rows; i++ ) {

		printf ( _us, i % scale );

		for ( std::uint32_t j = 0; j < m->n_cols; j++ )

			if ( ( std::uint32_t ) ( fabsf ( m->at ( i, j ) ) * precisionf + 0.5f ) == 0 ) {

				printf ( "%s", _zs );
			}

			else {

				printf ( _fs, m->at ( i, j ) );
			}

		printf ( "\n" );
	}

	printf ( "\n" );

	free ( _fs ); free ( _us ); free ( _zs ); free ( _es );
}
