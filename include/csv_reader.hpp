
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

#include <cassert>
#include <cctype> // tolower

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <sax/iostream.hpp> // <iostream> + nl, sp etc. defined...
#include <iterator>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

#include <sax/integer.hpp>
#include <sax/string_split.hpp>
#include <sax/zip.hpp>

#include "type_traits.hpp"
#include "bimap.hpp"
#include "num_translate.hpp"


namespace detail {


enum class line_ending { win = 0, nix, mac, err };


line_ending newline_type ( std::ifstream & str_ ) {
    const std::streampos cur = str_.tellg ( );
    str_.seekg ( str_.beg );
    int ch = str_.get ( );
    line_ending le = line_ending::err;
    while ( str_.eof ( ) != ch ) {
        if ( '\r' == ch ) {
            if ( '\n' == str_.peek ( ) ) {
                le = line_ending::win;
                break;
            }
            le = line_ending::mac;
            break;
        }
        if ( '\n' == ch ) {
            le = line_ending::nix;
            break;
        }
        ch = str_.get ( );
    }
    str_.clear ( ); // Clear errors, for seekg to lapack_work correctly.
    str_.seekg ( cur );
    return le;
}


std::istream & safe_getline ( std::istream & str_, std::string & t_, std::streambuf * sb_ ) {
    // Assumes std::istream::sentry has been set up.
    // https://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
    t_.clear ( );
    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks, such as thread synchronization
    // and updating the stream state.
    while ( true ) {
        int c = sb_->sbumpc ( );
        switch ( c ) {
        case EOF: // Also handle the case when the last line has no line ending.
            if ( t_.empty ( ) )
                str_.setstate ( std::ios::eofbit );
            return str_;
        case '\n':
            return str_;
        case '\r':
            if ( '\n' == sb_->sgetc ( ) )
                sb_->sbumpc ( );
            return str_;
        default:
            t_ += ( char ) c;
        }
    }
}


std::istream & safe_getline ( std::istream & str_, std::string & t_ ) {
    std::istream::sentry se ( str_, true );
    std::streambuf * sb = str_.rdbuf ( );
    return safe_getline ( str_, t_, sb );
}


bool safe_count_one_line ( std::istream & str_, std::streambuf * sb_ ) {
    // https://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks, such as thread synchronization
    // and updating the stream state.
    int c = sb_->sbumpc ( );
    switch ( c ) {
    case EOF: // Also handle the case when the last line has no line ending.
        str_.setstate ( std::ios::eofbit );
        return false;
    case '\r':
        if ( '\n' == sb_->sgetc ( ) )
            sb_->sbumpc ( );
    case '\n':
        return false;
    }
    while ( true ) {
        c = sb_->sbumpc ( );
        switch ( c ) {
        case '\r':
            if ( '\n' == sb_->sgetc ( ) )
                sb_->sbumpc ( );
        case EOF: // Also handle the case when the last line has no line ending.
        case '\n':
            return true;
        }
    }
}


std::uintptr_t safe_count_lines ( std::istream & str_, std::streambuf * sb_ ) {
    // Assumes std::istream::sentry has been set up.
    const std::streampos cur = str_.tellg ( );
    str_.seekg ( str_.beg );
    std::uintptr_t count = 0;
    while ( not ( str_.eof ( ) ) )
        count += safe_count_one_line ( str_, sb_ );
    str_.clear ( ); // Clear errors, for seekg to lapack_work correctly.
    str_.seekg ( cur );
    return count;
}


std::uintptr_t safe_count_lines ( std::istream & str_ ) {
    std::istream::sentry se ( str_, true );
    std::streambuf * sb = str_.rdbuf ( );
    return safe_count_lines ( str_, sb );
}

}



enum class in_out { in = 0, out };

namespace detail {
template<typename real>
[[ nodiscard ]] real string_to_real ( const std::string & s_ ) noexcept {
    real r;
    std::from_chars ( s_.data ( ), s_.data ( ) + s_.length ( ), r, std::chars_format::fixed );
    return r;
}
}

template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
struct csv_t {

    using id = std::uint32_t;

    csv_t ( ) = delete;
    csv_t ( const in_out io_ ) : m_in_out ( io_ ) { s_id++; }
    csv_t ( const char io_ ) : m_in_out ( 'i' == io_ ? in_out::in : in_out::out ) { s_id++; }

    real operator ( ) ( const std::string & s_ ) {
        return detail::string_to_real<real> ( s_ );
    }

    in_out m_in_out;

    private:

    real string_to_real ( std::string && s_ ) {
        // Get index from string, either construct and return it or find and return it.
        const auto it = s_table.template get<bm::from> ( ).find ( s_ );
        if ( it == std::end ( s_table ) || s_table.empty ( ) ) {
            real rv = ( real ) s_table.size ( );
            s_table.emplace ( std::move ( s_ ), rv );
            return rv;
        }
        return it->second;
    }

    static id s_id;
    static std::map<id, bm::bimap<std::string, real>> s_table;
};

template<typename real, typename sfinae>
typename csv_t<real, sfinae>::id csv_t<real, sfinae>::s_id = id { 0 };

template<typename real, typename sfinae>
std::map<typename csv_t<real, sfinae>::id, bm::bimap<std::string, real>> csv_t<real, sfinae>::s_table;


template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
struct csv_real {
    csv_real ( const in_out io_ ) : m_in_out ( io_ ) { }
    csv_real ( const char io_ ) : m_in_out ( 'i' == io_ ? in_out::in : in_out::out ) { }
    real operator ( ) ( const std::string & s_ ) {
        return detail::string_to_real<real> ( s_ );
    }
    in_out m_in_out;
    std::pair<in_out, std::size_t> vars ( ) const noexcept { // Find out no vars after all the samples have been read.
        return { m_in_out, std::size_t { 1 } };
    }
    private:
};


template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
struct csv_string {
    csv_string ( const in_out io_ ) : m_in_out ( io_ ) { }
    csv_string ( const char io_ ) : m_in_out ( 'i' == io_ ? in_out::in : in_out::out ) { }
    real operator ( ) ( std::string && s_ ) {
        // Get index from string, either construct and return it or find and return it.
        const auto it = s_table.template get<bm::from> ( ).find ( s_ );
        if ( it == std::end ( s_table ) || s_table.empty ( ) ) {
            real rv = ( real ) s_table.size ( );
            s_table.emplace ( std::move ( s_ ), rv );
            return rv;
        }
        return it->second;
    }
    in_out m_in_out;
    std::pair<in_out, std::size_t> vars ( ) const noexcept { // Find out no vars after all the samples have been read.
        return { m_in_out, sax::iLog<3> ( s_table.size ( ) - std::size_t { 1 } ) + std::size_t { 1 } };
    }
    private:
    static bm::bimap<std::string, real> s_table;
};

template<typename real, typename sfinae>
bm::bimap<std::string, real> csv_string<real, sfinae>::s_table;


template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
using csv_type = std::variant<std::monostate, csv_real<real, sfinae>, csv_string<real, sfinae>>;



template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
class csv_reader {

    std::istream & m_stream;
    std::istream::sentry m_sentry;
    std::streambuf * const m_stream_buffer;
    const index m_line_count;
    std::string m_line;
    index m_size = 0;
    std::pair<index, index> m_in_out_count { 0, 0 };
    std::vector<csv_type<real, sfinae>> m_types;
    std::vector<real> m_data;

    public:

    csv_reader ( ) = delete;
    csv_reader ( std::istream & stream_ ) :
        m_stream ( stream_ ),
        m_sentry ( m_stream, true ),
        m_stream_buffer ( m_stream.rdbuf ( ) ),
        m_line_count ( detail::safe_count_lines ( m_stream, m_stream_buffer ) ),
        m_types ( read_header ( ) ) {
        m_data.reserve ( m_types.size ( ) );
    }

    real operator [ ] ( const index index ) const {
        return m_data [ index ];
    }

    index length ( ) const noexcept {
        return m_size;
    }

    index lines ( ) const noexcept {
        return ( index ) m_line_count;
    }

    index input_size ( ) const noexcept {
        return ( index ) m_in_out_count.first;
    }

    index output_size ( ) const noexcept {
        return ( index ) m_in_out_count.second;
    }

    index input_bias_output_size ( ) const noexcept {
        return ( index ) m_in_out_count.first + 1 + m_in_out_count.second;
    }

    std::pair<index, index> dimensions ( ) const noexcept {
        return { lines ( ) - 1, ( index ) m_in_out_count.first + m_in_out_count.second }; // needs fixing
    }

    private:

    bool is_valid ( const std::string & s_ ) const noexcept {
        return
            "-" == s_ or
            "fi" == s_ or
            "fo" == s_ or
            "if" == s_ or
            "of" == s_ or
            "si" == s_ or
            "so" == s_ or
            "is" == s_ or
            "os" == s_;
    }

    void advance_to_header ( ) {
        int ch = m_stream_buffer->sgetc ( );
        while ( '@' != ch and EOF != ch ) {
            ch = m_stream_buffer->snextc ( );
        }
        if ( EOF == ch ) {
            std::cerr << "advance_to_header: no header line found.\n";
            exit ( 0 );
        }
    }

    void count_input_output ( ) {
        for ( auto & type : m_types ) {
            if ( std::holds_alternative<csv_real<real, sfinae>> ( type ) ) {
                if ( in_out::in == std::get<csv_real<real, sfinae>> ( type ).m_in_out ) {
                    ++m_in_out_count.first;
                }
                else {
                    ++m_in_out_count.second;
                }
            }
            else if ( std::holds_alternative<csv_string<real, sfinae>> ( type ) ) {
                if ( in_out::in == std::get<csv_string<real, sfinae>> ( type ).m_in_out ) {
                    ++m_in_out_count.first;
                }
                else {
                    ++m_in_out_count.second;
                }
            }
        }
    }

    std::vector<csv_type<real, sfinae>> read_header ( ) {

        // Header Line Grammar.
        // ...
        // @ - starts a header line.
        // - - ignores the column.
        // f - converts string in column to float and returns NAN iff it cannot be converted.
        // s - reads in a string.
        // i - input variable.
        // o - output variable.
        // ...
        // example:
        //
        //  @        -    fi    fi    fi    fi    fi    if    fi    fi   so
        //  ADH3_YEAST  0.51  0.51  0.52  0.51  0.50  0.00  0.54  0.22  MIT
        //  ADH4_YEAST  0.59  0.45  0.53  0.19  0.50  0.00  0.59  0.27  CYT
        // ...

        const auto cur = m_stream.tellg ( );

        // Read and parse header line...

        advance_to_header ( );
        detail::safe_getline ( m_stream, m_line, m_stream_buffer );
        std::vector<std::string> codes = sax::string_split ( m_line, "@", " ", ",", "\t" );

        // Construct types vector...

        std::vector<csv_type<real, sfinae>> types;

        index void_ctr = 0;

        for ( auto & code : codes ) {
            assert ( is_valid ( code ) );
            if ( 'f' == std::tolower ( code.front ( ) ) )
                types.emplace_back ( csv_real<real, sfinae> ( std::tolower ( code.back ( ) ) ) );
            else if ( 'f' == std::tolower ( code.back ( ) ) )
                types.emplace_back ( csv_real<real, sfinae> ( std::tolower ( code.front ( ) ) ) );
            else if ( 's' == std::tolower ( code.front ( ) ) )
                types.emplace_back ( csv_string<real, sfinae> ( std::tolower ( code.back ( ) ) ) );
            else if ( 's' == std::tolower ( code.back ( ) ) )
                types.emplace_back ( csv_string<real, sfinae> ( std::tolower ( code.front ( ) ) ) );
            else {
                ++void_ctr;
                types.emplace_back ( );
            }
        }
        m_size = types.size ( ) - void_ctr;
        count_input_output ( );
        m_stream.clear ( ); // Clear errors, for seekg to lapack_work correctly.
        m_stream.seekg ( cur );
        return types;
    }

    public:

    std::tuple<std::vector<std::pair<in_out, std::size_t>>, index, index> columns ( ) const noexcept {
        std::tuple<std::vector<std::pair<in_out, std::size_t>>, index, index> v;
        for ( auto & type : m_types ) {
            if ( std::holds_alternative<csv_real<real, sfinae>> ( type ) )
                std::get<0> ( v ).emplace_back ( std::get<csv_real<real, sfinae>> ( type ).vars ( ) );
            else if ( std::holds_alternative<csv_string<real, sfinae>> ( type ) )
                std::get<0> ( v ).emplace_back ( std::get<csv_string<real, sfinae>> ( type ).vars ( ) );
        }
        std::get<1> ( v ) = index { 0 }, std::get<1> ( v ) = index { 0 };
        for ( auto & cols : std::get<0> ( v ) ) {
            if ( in_out::in == cols.first )
                std::get<1> ( v ) += cols.second;
            else
                std::get<2> ( v ) += cols.second;
        }
        return v;
    }

    void read_line ( ) {
        char ch = 0;
        do {
            if ( m_stream.eof ( ) )
                return;
            detail::safe_getline ( m_stream, m_line, m_stream_buffer );
            ch = m_line [ m_line.find_first_not_of ( " " ) ];
        } while ( '@' == ch || '#' == ch || '\n' == ch || '\r' == ch ); // Skip header, comment and empty lines.
        m_data.clear ( );
        std::vector<std::string> values = sax::string_split ( m_line, " ", ",", "\t" );
        // Loop over columns.
        for ( auto [ type, value ] : sax::zip ( m_types, values ) ) {
            if ( std::holds_alternative<csv_real<real, sfinae>> ( type ) ) {
                m_data.emplace_back ( std::get<csv_real<real, sfinae>> ( type ) ( value ) );
            }
            else if ( std::holds_alternative<csv_string<real, sfinae>> ( type ) ) {
                m_data.emplace_back ( std::get<csv_string<real, sfinae>> ( type ) ( std::move ( value ) ) );
            }
        }
    }
};
