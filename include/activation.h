
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

// Activation Functions...

#include <cmath>

#include <algorithm>

#include "linear_algebra_float.h"
#include "type_traits.hpp"


enum class activation_function { unipolar, bipolar, rectifier };


template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
inline real activation_unipolar ( const real net, const real alpha ) noexcept {

    return real { 1 } / ( real { 1 } + std::exp ( -alpha * net ) );
}

template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
inline real activation_unipolar_derivative ( const real oput, const real alpha ) noexcept {

	return oput * ( alpha - alpha * oput );
}


template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
inline real activation_bipolar ( const real net, const real alpha ) noexcept {

	return real { 2 } / ( real { 1 } + std::exp ( - real { 2 } * alpha * net ) ) - real { 1 };
}

template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
inline real activation_bipolar_derivative ( const real oput, const real alpha ) noexcept {

	return alpha - ( alpha * oput * oput );
}


template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
inline real activation_rectifier ( const real net, const real alpha ) noexcept {

    return std::max ( net, real { 0 } );
}

template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
inline real activation_rectifier_derivative ( const real oput, const real alpha ) noexcept {

    return ( real ) ( oput > real { 0 } );
}


template<typename real, typename sfinae = typename std::enable_if<is_real<real>::value>::type>
inline real activation_softargmaxf ( const real x, const real y ) noexcept {

	return x < y ? y + std::log1p ( std::exp ( x - y ) ) : x + std::log1p ( std::exp ( y - x ) );
}


/*
static double logSumOfExponentials(double[] xs) {
        if (xs.length == 1) return xs[0];
        double max = maximum(xs);
        double sum = 0.0;
        for (int i = 0; i < xs.length; ++i)
            if (xs[i] != Double.NEGATIVE_INFINITY)
                sum += java.lang.Math.exp(xs[i] - max);
        return max + java.lang.Math.log(sum);
    }
*/
