
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

#include <utility>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>


namespace bm {

// Tags for accessing both sides of a bidirectional map...

struct from { };
struct to { };

namespace detail {

template<typename FromType, typename ToType>
struct bidirectional_map {

    using value_type = std::pair<FromType, ToType>;

    // A bidirectional map can be simulated as a multi_index_container of 
    // pairs of (FromType,ToType) with two unique indices, one for each 
    // member of the pair...

    using type = boost::multi_index::multi_index_container <

        value_type,
        boost::multi_index::indexed_by<
        boost::multi_index::ordered_unique<
        boost::multi_index::tag<from>, boost::multi_index::member<value_type, FromType, &value_type::first>>,
        boost::multi_index::ordered_unique<
        boost::multi_index::tag<to>, boost::multi_index::member<value_type, ToType, &value_type::second>>
    >>;
};

}

template<typename FromType, typename ToType>
using bimap = typename detail::bidirectional_map<FromType, ToType>::type;

}
