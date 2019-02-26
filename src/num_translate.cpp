// Package: numeric system translation
// Description: https://github.com/ans-hub/num_translate/
// Author: Anton Novoselov, 2017
// File: implementation of numeric translator functions

#include "num_translate.hpp"

namespace nt {

  // Represents exceptions

Exception::Exception(const Str& what_arg)
: std::logic_error(what_arg)
{ }

int conv_radix_digit_to_dec(char digit)
{
  int sub {0};    // offset from '9' in ascii
  if ((digit >= '0') && (digit <= '9'))
    sub = 0;
  else if ((digit >= 'A') && (digit <= 'Z'))
    sub = 7;
  else if ((digit >= 'a') && (digit <= 'z'))
    sub = 13;
  else
    throw Exception("conv_radix_digit_to_dec(): substitution not found");
  return static_cast<int>(digit - sub - kZeroChar);
}

char conv_dec_to_radix_digit(int dec)
{
  int add {0};    // offset from '9' in ascii
  if ((dec >= 0) && (dec <= 9))               // 0-9
    add = 0;
  else if ((dec >= 10) && (dec <= 35))        // A-Z
    add = 7;
  else if ((dec >= 36) && (dec <= 62))        // a-z
    add = 13;
  else 
    throw Exception("conv_dec_to_radix_digit(): substitution not found");    
  return static_cast<char>(dec + add + kZeroChar);
}

Str convert_n_to_dec(const Str& num, int base)
{
  if (base < kMinRadix || base > kMaxRadix)
    throw Exception("convert_n_to_dec(): base overflow");

  lint result {0};
  lint mp {1};
  for (auto it = num.rbegin(); it != num.rend(); ++it) {
    result += conv_radix_digit_to_dec(*it) * mp;
    mp *= base;
  }
  return std::to_string(result);
}

Str convert_dec_to_n(const Str& num, int base)
{
  lint dec {};
  try {
    dec = std::stoll(num);
  } catch (...) {
    throw Exception("convert_dec_to_n(): decimal overflow");
  }
  if (base < kMinRadix || base > kMaxRadix)
    throw Exception("convert_dec_to_n(): base overflow");

  Str result {};
  result.reserve(255);
  do {
    result.push_back(conv_dec_to_radix_digit(dec % base));
  } while (dec /= base);
  std::reverse(result.begin(), result.end());
  return result;
}

Str convert_dec_to_n ( lint dec, int base ) {
    
    Str result { };
    result.reserve ( 255 );
    do {
        result.push_back ( conv_dec_to_radix_digit ( dec % base ) );
    } while ( dec /= base );
    // std::reverse ( result.begin ( ), result.end ( ) );
    return result;
}

Str convert_a_to_b(const Str& num, int src, int dest)
{
  if (src < kMinRadix || src > kMaxRadix)
    throw Exception("convert_a_to_b(): src base overflow");
  else if (dest < kMinRadix || dest > kMaxRadix)
    throw Exception("convert_a_to_b(): dest base overflow");
  else if (src == dest)
    return num;
  else {
    Str tmp = convert_n_to_dec(num, src);
    return convert_dec_to_n(tmp, dest);
  }
}

lint fast_pow(lint base, lint exp)
{
  if (exp < 0)
    throw Exception("Fast_pow: negative exp is not supported");   

  lint res {1};
  while(exp) {
    if (exp & 1)
      res *= base;
    exp >>= 1;
    base *= base;
  }
  return res;
}

} // namespace nmtrs
