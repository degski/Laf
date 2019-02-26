
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

#include <filesystem>

namespace fs = std::filesystem;

#include <fstream>
#include <string>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <lz4stream.hpp>


template<typename T>
void saveToFile ( const T & t_, std::string && file_name_ ) noexcept {

    std::ofstream compressed_ostream ( fs::current_path ( ) / ( file_name_ + std::string ( ".lz4cereal" ) ), std::ios::binary );
    LZ4OutputStream lz4_ostream ( compressed_ostream );

    cereal::BinaryOutputArchive archive ( lz4_ostream );

    archive ( t_ );

    lz4_ostream.flush ( );
    compressed_ostream.flush ( );

    lz4_ostream.close ( );
    compressed_ostream.close ( );
}

template<typename T>
void loadFromFile ( T & t_, std::string && file_name_ ) noexcept {

    std::ifstream compressed_istream ( fs::current_path ( ) / ( file_name_ + std::string ( ".lz4cereal" ) ), std::ios::binary );
    LZ4InputStream lz4_istream ( compressed_istream );

    cereal::BinaryInputArchive archive ( lz4_istream );

    archive ( t_ );

    compressed_istream.close ( );
}
