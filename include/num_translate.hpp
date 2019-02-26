// Library: numeric system translation (v0.41)
// Description: https://github.com/ans-hub/num_translate/
// Author: Anton Novoselov, 2017
// File: interface of numeric translator functions
// Terms: 
//  digit (char)    - the symbol in radix N (Eg: N=16, uses 0..9..A..F) )
//  number (string) - the number in radix N (Eg: N=16, number = 9F) 
//  radix_range     - kMinRadix..kMaxRadix
//  radix_digits    - 0..9..A..Z..a..z

#ifndef NUM_TRANSLATE_H
#define NUM_TRANSLATE_H

#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nt {
  
// Aliases and constants of functions below

using Str   = std::string;
using lint  = long long; 

constexpr int kMinRadix = 2;
constexpr int kMaxRadix = 62;
constexpr int kZeroChar = 48;

// Represents exceptions of functions below

struct Exception : public std::logic_error
{
  Exception(const Str&);
};

// Converts radix N representation of 'digit` in radix N to decimal
// Example: f('1') returns 1, f('F') returns 15

int conv_radix_digit_to_dec(char);

// Converts decimal representation of 'digit' in radix N to 'digit' in radix N 
// Example: f(15) = 'F'; f(16) = 'G'; f(37) = 'a'

char conv_dec_to_radix_digit(int);

// Covert number from radix N to decimal
// Example: f("6F",16) = 90; f("AB",20) = 211

Str convert_n_to_dec(const Str& num, int base);

// Convert decimal to number radix N
// Example: f(10,16) = "A"; f(211,20) = "AB"

Str convert_dec_to_n(const Str& num, int base);
Str convert_dec_to_n(lint num, int base);

// Convert number from radix A to radix B notation
// Example: f("32",4,12) = "3C"

Str convert_a_to_b(const Str& num, int src, int dest);

// Returns power of number
// Example: f(10,2) = 100

lint fast_pow(lint base, lint exp);

}  // nmtrs

#endif // NUM_TRANSLATE_H
