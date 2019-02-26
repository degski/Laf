
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

#include <cassert>
#include <cctype> // tolower
#include <cstdint>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <sax/iostream.hpp>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

#if !defined(NDEBUG)
#define BOOST_MULTI_INDEX_ENABLE_INVARIANT_CHECKING
#define BOOST_MULTI_INDEX_ENABLE_SAFE_MODE
#endif

#include <sax/autotimer.hpp>
#include <sax/string_split.hpp>

#include "activation.h"
#include "linear_algebra_float.h"
#include "fully_connected_ann.h"
#include "multi_layer_perceptron.h"
#include "simplified_fully_connected_ann.h"

#include "csv_reader.hpp"
#include "matrix.hpp"
#include "bimap.hpp"
#include "fccn.hpp"


// inputs: value
// output: gray
// Convert a value to a Gray code with the given base.
// Iterating through a sequence of values would result in a sequence
// of Gray codes in which only one digit changes at a time.

template<typename T, T base, typename sfinae = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
std::vector<T> to_gray ( T value ) {

    // https://en.wikipedia.org/wiki/Gray_code

    const T digits = sax::iLog<base> ( value ) + T { 1 };

    std::vector<T> baseN ( digits, T { 0 } ); // Stores the ordinary base-N number, one digit per entry...
    T i; // The loop variable...

    // Put the normal baseN number into the baseN array. For base 10, 109
    // would be stored as [9,0,1]...

    for ( i = T { 0 }; i < digits; i++ ) {

        baseN [ i ] = value % base;
        value = value / base;
    }

    // Convert the normal baseN number into the Gray code equivalent. Note that
    // the loop starts at the most significant digit and goes down...

    value = T { 0 }; // value is (has become) the shift...

    std::vector<T> gray ( digits, T { 0 } );

    while ( i-- ) {

        // The gray digit gets shifted down by the sum of the higher digits...

        gray [ i ] = ( baseN [ i ] + value ) % base;
        value += base - gray [ i ]; // Subtract from base so shift is positive...
    }

    std::reverse ( gray.begin ( ), gray.end ( ) );

    return gray;
}


int main3225464 ( ) {

    const auto v = to_gray<std::uint32_t, 7> ( 1900u );

    for ( auto i : v ) {

        std::cout << i << " ";
    }

    // std::cout << nl;

    return 0;
}



#include <hike/hike_fi_local_search.h>
#include <hike/hike_vns.h>

int main ( ) {

    using real = double;
    using index = std::uint32_t;

    fccn<real, index> * fcc = fccn<real, index>::construct ( 16, 8, 3, activation_function::unipolar, 1.0 );
    fcc->read_in_out ( "y:/repos/laf/data/yeast_new.csv", activation_function::unipolar );

    return 0;

    vns_loss_function<real, index> loss_function = { fcc };
    vns_solution<real> step_solution ( fcc->wgts, 0.1 );
    vns_local_search<real, index> local_search ( loss_function, step_solution );
    vns<real, index> vns ( local_search, 5 );

    vns_solution<real> solution ( fcc->wgts );
    std::memcpy ( ( void* ) solution.data ( ), ( void* ) fcc->wts_data, fcc->wgts * sizeof ( real ) );

    bool optimized;
    vns_solution<real> optimized_solution = vns.optimize ( solution, optimized );

    std::cout << loss_function ( optimized_solution ) << nl;

    /*

    fcc->construct_pop ( 64u, 64u / 2u );

    while ( true ) {

        std::cout << "error " << fcc->regenerate_pop ( ) << nl;
    }

    exit ( 0 );

    */



    real ase1 = 0;

    {
        sax::AutoTimer timer;

        fcc->levenberg_marquardt_train1 ( );
    }

    std::cout << "ase1 " << ase1 << nl;



    return 0;
}

/*

TEST_CASE ( "VNS example" ) {

// Solution is a 3D integer vector. It can be of
// any type and size...

using Solution = std::array<double, 3>;

// Loss function return an integer scalar. It can
// be of any type...

struct LossFunction {

Solution targetSolution;

double operator ( ) ( const Solution & solution ) const {

double loss = 0.0;

for ( std::size_t i = 0; i < solution.size ( ); ++i ) {

loss += std::abs ( solution [ i ] - targetSolution [ i ] );
}

return loss;
}
};

// Loss function returns the Manhattan distance
// between the target solution and the given one...

Solution targetSolution { { 2.0, 5.0, -10.0 } };
LossFunction lossFunction = { targetSolution };

// VNS uses first improvement (first descent) local
// search. Best improvement (highest descent) local
// search can be used too..

using LocalSearch = hike::FILocalSearch<Solution, LossFunction>;

// Candidate solutions are generated adding and sub-
// tracting the parameters of this solution to the
// given one...

Solution stepSolution { { 0.1, 0.1, 0.1 } };

// Declare local search object...

LocalSearch localSearch ( lossFunction, stepSolution );

// Declare VNS object with a maximum neighborhood (kmax) of 5...

hike::VNS<Solution, LocalSearch> vns ( localSearch, 5 );

// Optimize a solution...

Solution solution { { 15.0, -7.0, 22.0 } };

bool optimized;
Solution optimizedSolution = vns.optimize ( solution, optimized );

// The optimized solution should be equals to the target one...

REQUIRE ( optimized );
REQUIRE ( lossFunction ( optimizedSolution ) < 0.0001 );

std::cout << '\n';

for ( auto v : solution ) {

std::cout << v << ' ';
}

std::cout << '\n';
}

*/


int32_t main325455 ( ) {

    linear_algebra_float_seed ( 15802527 ); // first used for testing...
    // linear_algebra_float_seed ( 566451230 );

    using real = double;
    using index = std::uint32_t;

    fds *ds = linear_algebra_float_data_set_malloc ( yeast );

    fccn<real, index> * fcc = fccn<real, index>::construct ( 15, ds->n_iput, ds->n_oput, activation_function::unipolar, 0.7f );

    // fccn<real, index> * fcc = fccn<real, index>::construct ( 4, 3, 1, activation_function::unipolar, 1.0f );

    std::cout << "fccn size: " << sizeof ( fccn<real, index> ) << " weights: "<< fcc->wgts << " iput " << ds->n_iput << " oput " <<  ds->n_oput << nl;

    fcc->tr_ibo = matrix<real, index>::construct_zero ( ds->training.n_patt, fcc->iput + 1 + fcc->nrns );
    fcc->tr_ibo_data = fcc->tr_ibo->data ( );
    for ( index i = 0; i < ds->training.iput->n_elem; ++i ) { fcc->tr_ibo_data [ i ] = ( real ) ds->training.iput->v [ i ]; }

    // std::cout << fcc->tr_ibo << nl;

    fcc->tr_out = matrix<real, index>::construct ( ds->training.n_patt, fcc->oput );
    fcc->tr_out_data = fcc->tr_out->data ( );
    for ( index i = 0; i < ds->training.oput->n_elem; ++i ) { fcc->tr_out_data [ i ] = ( real ) ds->training.oput->v [ i ]; }

    // std::cout << fcc->tr_out << nl;

    // exit ( 0 );

    fcc->tr_patt = fcc->tr_ibo->rows;

    fcn *cn = fully_connected_cascade_malloc ( 15, ds->n_iput, ds->n_oput, activation_function::unipolar, 0.7f );

    fcc->construct_pat ( );

    // fcc->copy_wts ( cn->wts ); // This is just plugged in to have the same weights...

    real ase1 = 0;

    {
        sax::AutoTimer timer;

        ase1 = fcc->levenberg_marquardt_train1 ( );

        // ase1 = fully_connected_cascade_train ( cn, ds );
    }

    std::cout << "ase1 " << ase1 << nl;
    /*
    fcc->copy_wts ( cn->wts ); // This is just plugged in to have the same weights...

    real ase2 = 0;

    {
        at::AutoTimer timer;

        ase2 = fcc->levenberg_marquardt_train2 ( );
    }

    std::cout << "ase2 " << ase2 << nl;
    */
    return 0;
}


std::streampos lastLinePosition ( std::ifstream & str_ ) {

    const std::streampos cur = str_.tellg ( );

    std::streamoff off = -1;

    // Correct line count for messy (lack of or several) nl's at the end...

    while ( true ) {  // go to one spot before the EOF...

        str_.seekg ( off, std::ios_base::end );

        if ( '\n' != str_.get ( ) ) {

            break;
        }

        --off;
    }

    bool keep_looping = true;

    while ( keep_looping ) {

        char ch;
        str_.get ( ch );// Get current byte's data...

        if ( 2 > ( lapack_int ) str_.tellg ( ) ) { // If the data was at or before the 0th byte...

            str_.seekg ( 0 ); // The first line is the last line...
            keep_looping = false; // So stop there...
        }

        else if ( '\n' == ch ) { // If the data was a newline...

            keep_looping = false;// Stop at the current position...
        }

        else { // If the data was neither a newline nor at the 0 byte...

            str_.seekg ( -2, std::ios_base::cur ); // Move to the front of that data, then to the front of the data before it...
        }
    }

    const std::streampos last_line_position = str_.tellg ( );

    str_.clear ( ); // Clear errors, for seekg to lapack_work correctly...
    str_.seekg ( cur );

    return last_line_position;
}

using index = std::uint32_t;

index no_digits ( const index max_ ) {

    // return std::floor ( std::log10 ( max_ ) / std::log10 ( 3 ) + 1 );

    return sax::iLog<3> ( 9u );
}

// #include "num_translate.hpp"

int main243324 ( ) {

    for ( index i = 0; i < 20; ++i ) {

        // std::cout << i << " " << sax::iLog<3> ( i ) << " " << nt::convert_dec_to_n ( i, 3 ) << nl;
    }

    return 0;
}

// ADT1_YEAST  0.58  0.61  0.47  0.13  0.50  0.00  0.48  0.22  MIT


int main46576 ( ) {

    const auto activation_function_ = activation_function::unipolar;

    using real = float;
    using pointer = real*;
    using index = std::uint32_t;

    std::ifstream file ( "yeast_new.csv", std::ios_base::binary | std::ios_base::in );

    if ( file.is_open ( ) ) {

        csv_reader<real, index> reader ( file );

        matrix<real, index> m ( reader.dimensions ( ), matrix_creation::zero );

        const index patt = m.rows;
        index line = index { 0 };

        for ( index l = index { 1 }; l < reader.lines ( ); ++l ) {

            reader.read_line ( );

            for ( index i = index { 0 }, j = index { 0 }; j < reader.length ( ); ++i ) {

                m.at ( line, i ) = reader [ j++ ];
            }

            ++line;
        }

        const auto [ data, i_size, o_size ] = reader.columns ( );

        matrix<real, index> *i_put = matrix<real, index>::construct_zero ( patt, i_size + index { 1 } );
        matrix<real, index> *o_put = matrix<real, index>::construct_zero ( patt, o_size );

        pointer f = m.data ( ), i = i_put->data ( ), o = o_put->data ( );

        for ( const auto & cols : data ) {

            if ( in_out::in == cols.first ) {

                std::memcpy ( ( void* ) i, ( void* ) f, sizeof ( real ) * patt );

                if ( index { 1 } < cols.second ) {

                    for ( pointer u = i, l = u + patt; u < l; ++u ) {

                        auto code = to_gray<index, 3> ( ( index ) *u );

                        pointer a = u;

                        for ( ; index { 0 } < code.size ( ); a += patt ) {

                            if ( activation_function_ == activation_function::unipolar ) {

                                *a = ( real ) code.back ( );
                                *a *= real { 1 } / real { 2 };
                            }

                            else {

                                *a = ( real ) ( ( int ) code.back ( ) - 1 );
                            }

                            code.pop_back ( );
                        }

                        if ( activation_function_ == activation_function::bipolar ) {

                            for ( const pointer la = u + cols.second * patt; a < la; a += patt ) {

                                *a = real { -1 };
                            }
                        }
                    }
                }

                i += cols.second * patt;
                f += patt;
            }

            else { // If out...

                std::memcpy ( ( void* ) o, ( void* ) f, sizeof ( real ) * patt );

                if ( index { 1 } < cols.second ) {

                    for ( pointer u = o, l = u + patt; u < l; ++u ) {

                        auto code = to_gray<index, 3> ( ( index ) *u );

                        pointer a = u;

                        for ( ; index { 0 } < code.size ( ); a += patt ) {

                            if ( activation_function_ == activation_function::unipolar ) {

                                *a = ( real ) code.back ( );
                                *a *= real { 1 } / real { 2 };
                            }

                            else {

                                *a = ( real ) ( ( int ) code.back ( ) - 1 );
                            }

                            code.pop_back ( );
                        }

                        if ( activation_function_ == activation_function::bipolar ) {

                            for ( const pointer la = u + cols.second * patt; a < la; a += patt ) {

                                *a = real { -1 };
                            }
                        }
                    }
                }

                o += cols.second * patt;
                f += patt;
            }
        }

        for ( const pointer li = i + patt; i < li; ++i ) {

            *i = real { 1 };
        }

        std::cout << i_put << nl;
        std::cout << o_put << nl;

        file.close ( );
    }

    return 0;
}



int main32543655 ( ) {

    const int N = 3;
    const int K = 16;

    int n [ K + 1 ]; // The maximum for each digit...
    int g [ K + 1 ]; // The Gray code...
    int u [ K + 1 ]; // +1 or -1 ...

    int i, j, k;

    for ( i = 0; i <= K; ++i ) {

        g [ i ] = 0; u [ i ] = 1;
        n [ i ] = N;
    }

    while ( 0 == g [ K ] ) {

        printf ( "(" );
        for ( j = K - 1; j >= 0; j-- ) printf ( " %d", g [ j ] );
        printf ( ")\n" );

        i = 0; // Enumerate next Gray code...
        k = g [ 0 ] + u [ 0 ];

        while ( ( k >= n [ i ] ) or ( k < 0 ) ) {

            u [ i ] = -u [ i ];
            ++i;
            k = g [ i ] + u [ i ];
        }

        g [ i ] = k;
    }

    return 0;
}


int main354645 ( ) {

    std::ifstream file ( "yeast1.csv" );

    if ( file.is_open ( ) ) {

        const std::streampos llp = lastLinePosition ( file );

        csv_reader<float, std::uint32_t> reader ( file );

        while ( llp != file.tellg ( ) ) {

            reader.read_line ( );

            std::cout << reader [ 8 ] << "\n";
        }

        file.close ( );
    }

    return 0;
}


/*
class CSVIterator {

    public:

    typedef std::input_iterator_tag iterator_category;
    typedef csv_reader value_type;
    typedef std::size_t difference_type;
    typedef csv_reader * pointer;
    typedef csv_reader & reference;

    CSVIterator ( std::istream &str_ ) :m_str ( str_.good ( ) ? &str_ : nullptr ) { ++( *this ); }
    CSVIterator ( ) :m_str ( nullptr ) { }

    // Pre Increment...

    CSVIterator &operator ++ ( ) { if ( m_str ) { ( *m_str ) >> m_row; m_str = m_str->good ( ) ? m_str : nullptr; }return *this; }

    // Post increment...

    CSVIterator operator ++ ( lapack_int ) { CSVIterator tmp ( *this ); ++( *this ); return tmp; }
    csv_reader const &operator * ( )   const { return m_row; }
    csv_reader const *operator -> ( )  const { return &m_row; }

    bool operator == ( CSVIterator const &rhs_ ) { return ( ( this == &rhs_ ) || ( ( this->m_str == nullptr ) && ( rhs_.m_str == nullptr ) ) ); }
    bool operator != ( CSVIterator const &rhs_ ) { return !( ( *this ) == rhs_ ); }

    private:

    std::istream *m_str;
    csv_reader m_row;
};


int main4565467578 ( ) {

    std::ifstream file ( "iris.csv" );

    std::cout << countLines ( file ) << std::endl;

    for ( CSVIterator loop ( file ); loop != CSVIterator ( ); ++loop ) {

        std::cout << "4th Element(" << ( *loop ) [ 3 ] << ")\n";
    };

    return 0;
}

*/

#include <set>
#include <utility>
#include <iostream>
#include <algorithm>


template<typename X, typename Y, typename x_less = std::less<X>, typename y_less = std::less<Y>>
class bimap1 {

    // https://stackoverflow.com/questions/21760343/is-there-a-more-efficient-implementation-for-a-bidirectional-map

    using key_type = std::pair<X, Y>;
    using value_type = std::pair<X, Y>;

    using iterator = typename std::set<key_type>::iterator;
    using const_iterator = typename std::set<key_type>::const_iterator;

    struct x_comp {

        bool operator ( ) ( X const & x1, X const & x2 ) const {

            return not ( x_less ( ) ( x1, x2 ) ) and not ( x_less ( ) ( x2, x1 ) );
        }
    };

    struct y_comp {

        bool operator ( ) ( Y const & y1, Y const & y2 ) const {

            return not ( y_less ( ) ( y1, y2 ) ) and not ( y_less ( ) ( y2, y1 ) );
        }
    };

    struct f_less {

        // Prevents lexicographical comparison for std::pair, so that
        // every .first value is unique as if it was in its own map...

        bool operator ( ) ( key_type const & lhs, key_type const & rhs ) const {

            return x_less ( ) ( lhs.first, rhs.first );
        }
    };

    /// Key and value type are interchangeable...

    std::set<std::pair<X, Y>, f_less> m_data;

    public:

    std::pair<iterator, bool> insert ( X const & x, Y const & y ) {

        auto it = find_right ( y );

        if ( it == cend ( ) ) { // every .second value is unique

            return m_data.emplace ( x, y );
        }

        return { it, false };
    }

    std::pair<iterator, bool> insert ( X && x, Y && y ) {

        auto it = find_right ( y );

        if ( it == cend ( ) ) { // every .second value is unique

            return m_data.emplace ( std::move ( x ), std::move ( y ) );
        }

        return { it, false };
    }

    std::pair<iterator, bool> insert ( X const & x, Y && y ) {

        auto it = find_right ( y );

        if ( it == cend ( ) ) { // every .second value is unique

            return m_data.emplace ( x, std::move ( y ) );
        }

        return { it, false };
    }

    std::pair<iterator, bool> insert ( X && x, Y const & y ) {

        auto it = find_right ( y );

        if ( it == cend ( ) ) { // every .second value is unique

            return m_data.emplace ( std::move ( x ), y );
        }

        return { it, false };
    }

    iterator find_left ( X const & val ) const noexcept {

        return m_data.find ( { val, Y ( ) } );
    }

    iterator find_right ( Y const & val ) const noexcept {

        return std::find_if ( cbegin ( ), cend ( ),

            [ & val ] ( key_type const & kt ) {

                return y_comp ( ) ( kt.second, val );
            }
        );
    }

    iterator end ( ) noexcept { return m_data.end ( ); }
    iterator begin ( ) noexcept { return m_data.begin ( ); }

    const_iterator cend ( ) const noexcept { return m_data.cend ( ); }
    const_iterator cbegin ( ) const noexcept { return m_data.cbegin ( ); }
};


template<typename X, typename Y, typename In>
void PrintBimapInsertion ( X const &x, Y const &y, In const &in ) {
    if ( in.second ) {
        std::cout << "Inserted element ("
            << in.first->first << ", " << in.first->second << ")\n";
    }
    else {
        std::cout << "Could not insert (" << x << ", " << y << ") because (" <<
            in.first->first << ", " << in.first->second << ") already exists\n";
    }
}


int main23346576587 ( ) {

    bimap1<std::string, int> mb;

    PrintBimapInsertion ( "A", 1, mb.insert ( "A", 1 ) );
    PrintBimapInsertion ( "A", 2, mb.insert ( "A", 2 ) );
    PrintBimapInsertion ( "b", 2, mb.insert ( "b", 2 ) );
    PrintBimapInsertion ( "z", 2, mb.insert ( "z", 2 ) );

    auto it1 = mb.find_left ( "A" );

    if ( it1 != mb.end ( ) ) {

        std::cout << std::endl << it1->first << ", " << it1->second << std::endl;
    }

    auto it2 = mb.find_right ( 2 );

    if ( it2 != mb.end ( ) ) {

        std::cout << std::endl << it2->first << ", " << it2->second << std::endl;
    }

    return 0;
}



template<typename pair>
struct reverse {

    using second_type = typename pair::first_type;
    using first_type = typename pair::second_type;

    second_type second;
    first_type first;
};

template<typename pair>
reverse<pair> & mutate ( pair & p ) {

    return reinterpret_cast<reverse<pair>&> ( p );
}

int main32543657 ( void ) {

    std::pair<double, int> p ( 1.34, 5 );

    std::cout << "p.first = " << p.first << ", p.second = " << p.second << std::endl;
    std::cout << "mutate(p).first = " << mutate ( p ).first << ", mutate(p).second = " << mutate ( p ).second << std::endl;

    return 0;
}



// A dictionary is a bidirectional map from strings to strings...

using dictionary = bm::bimap<std::string, std::string>;


int main3456354767869809 ( ) {

    dictionary d;

    // Fill up our microdictionary. first members Spanish, second
    // members English...

    d.emplace ( "hola", "hello" );
    d.emplace ( "adios", "goodbye" );
    d.emplace ( "rosa", "rose" );
    d.emplace ( "mesa", "table" );

    std::cout << "enter a word" << std::endl;
    std::string word;
    std::getline ( std::cin, word );

    // Search the queried word on the from index (Spanish)...

    auto it1 = d.get<bm::from> ( ).find ( word );

    if ( it1 != d.get<bm::from> ( ).end ( ) ) { // found...

    // The second part of the element is the equivalent in English...

        std::cout << word << " is " << it1->second << " in English." << std::endl;
    }

    else {

        // Word not found in Spanish, try our luck in English...

        auto it2 = d.get<bm::to> ( ).find ( word );

        if ( it2 != d.get<bm::to> ( ).end ( ) ) {

            std::cout << word << " is " << it2->first << " in Spanish." << std::endl;
        }

        else std::cout << "No such word in the dictionary." << std::endl;
    }

    return 0;
}

/*

#include <iostream>
#include <array>
#include <string>


namespace constexpr_xor_array {


namespace details {

template<::std::size_t...>
struct index_list { };

template<typename index_list, ::std::size_t>
struct appender { };

template<::std::size_t... left, ::std::size_t right>
struct appender<index_list<left...>, right> {
    typedef index_list<left..., right> type;
};

template<::std::size_t N>
struct make_index_list {
    typedef typename appender<typename make_index_list<N - 1>::type, N - 1>::type type;
};

template<>
struct make_index_list<0> {
    typedef index_list<> type;
};

}


template<typename T, ::std::size_t N>
class xor_array {
    private:
    template<typename>
    class xor_array_inner { };
    template<::std::size_t... Idxs>
    class xor_array_inner<constexpr_xor_array::details::index_list<Idxs...>> {
        public:
        template<::std::size_t M>
        constexpr xor_array_inner ( const T ( &arr ) [ M ] ) :
            m_value { encrypt_elem ( arr, M, Idxs, 0 )... } { }
        public:
        T operator[]( ::std::size_t const& idx ) const {
            return decrypt_elem ( idx, 0 );
        }
        private:
        constexpr static T encrypt_elem ( T const* const& arr, ::std::size_t const& M, ::std::size_t const& idx, ::std::size_t const& inner_idx ) {
            return inner_idx == sizeof ( T ) ? T { 0 } : encrypt_elem ( arr, M, idx, inner_idx + 1 ) | encrypt_byte ( ( ( idx < M ? arr [ idx ] : T { 0 } ) >> ( inner_idx * CHAR_BIT ) & 0xFF ), idx * sizeof ( T ) + inner_idx ) << ( inner_idx * CHAR_BIT );
        }
        constexpr static unsigned char encrypt_byte ( T const& byte, ::std::size_t const& idx ) {
            return static_cast<unsigned char>( ( static_cast<unsigned char>( byte & 0xFF ) ^ static_cast<unsigned char>( ( s_secret_random_xor_key + idx ) & 0xFF ) ) & 0xFF );
        }
        T decrypt_elem ( ::std::size_t const& idx, ::std::size_t const& inner_idx ) const {
            return inner_idx == sizeof ( T ) ? T { 0 } : decrypt_elem ( idx, inner_idx + 1 ) | decrypt_byte ( ( m_value [ idx ] >> ( inner_idx * CHAR_BIT ) & 0xFF ), idx * sizeof ( T ) + inner_idx ) << ( inner_idx * CHAR_BIT );
        }
        static unsigned char decrypt_byte ( T const& byte, ::std::size_t const& idx ) {
            return static_cast<unsigned char>( ( static_cast<unsigned char>( byte & 0xFF ) ^ static_cast<unsigned char>( ( s_secret_random_xor_key + idx ) & 0xFF ) ) & 0xFF );
        }
        private:
        constexpr static unsigned char const s_secret_random_xor_key = 42;
        private:
        ::std::array<T const, N> const m_value;
    };
    public:
    template<::std::size_t M>
    constexpr xor_array ( const T ( &arr ) [ M ] ) :
        m_inner ( arr ) { }
    public:
    constexpr ::std::size_t size ( ) const {
        return N;
    }
    T operator[]( ::std::size_t const& idx ) const {
        return m_inner [ idx ];
    }
    private:
    xor_array_inner<typename constexpr_xor_array::details::make_index_list<N>::type> const m_inner;
};


}


int main12236789 ( ) {
    static constexpr const constexpr_xor_array::xor_array<unsigned int, 100> hidden_arr_1 { { 0x12345678, 2, 3, 0x87654321 } };
    for ( ::std::size_t i = 0; i != hidden_arr_1.size ( ); ++i ) {
        ::std::cout << ::std::hex << hidden_arr_1 [ i ] << " ";
    }
    ::std::cout << ::std::endl;

    static constexpr const constexpr_xor_array::xor_array<char, 100> hidden_arr_2 { "MarekKnapek" };
    for ( ::std::size_t i = 0; i != hidden_arr_2.size ( ); ++i ) {
        ::std::cout << hidden_arr_2 [ i ];
    }
    ::std::cout << ::std::endl;

    static constexpr const constexpr_xor_array::xor_array<wchar_t, 100> hidden_arr_3 { L"MarekKnapek" };
    for ( ::std::size_t i = 0; i != hidden_arr_3.size ( ); ++i ) {
        ::std::wcout << hidden_arr_3 [ i ];
    }
    ::std::wcout << ::std::endl;

    return 0;
}

*/
