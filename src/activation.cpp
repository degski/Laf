
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

#include "activation.h"


#if 0

//------------------------------------

/* NEURON ACTIVATION FUNCTIONS */
//--------------------------------------
//unipolar sigmoid
double func::sigmoid(double x,double alpha)
{
    return(1.0/(1+exp(-alpha*x)));
}
//--------------------------------------
//bipolar sigmoid
double func::tansig(double x,double alpha)
{
   return(-1.0+2.0/(1+exp(-alpha*x)));
}
//-----------------------
// y = 2/pi * arctan(a * x)
double func::arctan(double x, double a)
{
  if(a <= 0)
  {
    cout << "the second argument to arctan must be positive\n";
    exit(-1);
  }
  double s = 2/M_PI * atan(a*x);
  return s;
}
//----------------------
// Gaussian function
// Date: 16/03/07
double func::gauss(double x, double sigma)
{
   return(exp(-x*x/(2*sigma*sigma)));
}
//--------------------------
//linear func function;y=a*x
double func::purelin(double x,double alpha=1)
{
   return(alpha*x);
}
//-------------
// tanh function
// Date: 31/05/07
double func::hyptan(double x, double a, double b)
{
  return (a * tanh(b*x));
}
//--------------------
// Thin-plate-spline function
// Date: 31/05/07
double func::tps(double x, double a)
{
  double s;
  s = a*a*x*x*log(a*x);
  return s;
}
//-------------------------------------
// Hyperbolic cosine function
// Date: 02/06/07
double func::hypcos(double x)
{
  return(cosh(x) - 1);
}
//-------------------------
// square of tansig function
//Date: 30 Sep. 2008 Tuesday
// Note that : y = (1-exp(-x))/(1+exp(-x))
// But this is not defined for x = -inf
// -------------------------------------------
double func::tansigsq(double x, double a=1)
{
  double z = a * x;

  double y = -1.0 + 2.0/(1+exp(-z));

  return (0.5*y*y);
}
//-----------------------------
double func::square(double x, double a = 1)
{
  double z = a * x;
  double y = z * z;

  return(y);
}

// ===============================

// Derivative of tansig function
double func::dtansig(double x,double alpha)
{
  // Note that x is the input Neuron.
  double y = tansig(x, alpha);

  return(0.5*alpha*(1+y)*(1-y));
}
//--------------------------
//Derivative of sigmoid function
double func::dsigmoid(double x,double alpha)
{
  // Note that x is the input to the neuron
  double y = sigmoid(x, alpha);
  return(alpha*y*(1-y));
}
//--------------------------
// Derivative of arctan
//
// dy/dx = 2a / (pi * (1+a^2x^2))
//
// Note that x is the input to the neuron
// ---------------------------------------
double func::darctan(double x, double a)
{
  double s = 2 * a / (M_PI*(1 + a*a*x*x));
  return s;
}
//----------------------------
//Derivative of Gaussian function
double func::dgauss(double x, double sigma)
{
  // Note that x is the input to the neuron
  // and y is the output of the neuron
  double y;
  y = gauss(x, sigma);
  return(-x*y/(sigma*sigma));
}
//--------------
//Derivative of linear function dy/dx=a
// x is the input to the neuron
double func::dlin(double x, double alpha=1)
{
  x = x;
  return(alpha);
}
//--------------
//Derivative of tanh function
//Date: 31/05/07
// x is the input to the neuron
double func::dhyptan(double x, double a, double b)
{
  double s;
  s = a * b * (1 - tanh(b*x) * tanh(b*x));

  return s;
}
//-------------------
//Derivative of tps function
// Date: 31/05/07
double func::dtps(double x, double a)
{
  double s, t;
  s = a * x;
  t = a * s * (1 + 2 * log(s));
  return t;
}
//-----------------------------------
// Derivative of cosh(x)
// Date: 02/06/07
double func::dhypcos(double x)
{
  return (sinh(x));
}
//--------------------------
double func::dtansigsq(double x, double a=1)
{
  //x is the input to dtansigsq function

  double z = a * x;
  double y = -1.0 + 2.0/(1+exp(-z));

  double s = 0.5 * a * y * (1+y) * (1-y);

  return(s);
}
//-----------------------------------
double func::dsquare(double x, double a=1)
{
  // x is the input to dsquare function

  double y = 2 * a * a * x;

  return(y);
}

#endif
