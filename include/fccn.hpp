
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


#include <algorithm>
#include <functional>
#include <sax/iostream.hpp> // <iostream> + nl, sp etc. defined.
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <sax/ska_sort.hpp>
#include <sax/autotimer.hpp>

#include <hike/hike_fi_local_search.h>
#include <hike/hike_vns.h>

#include "activation.h"
#include "type_traits.hpp"
#include "serialize.hpp"
#include "matrix.hpp"
#include "csv_reader.hpp"
#include "num_translate.hpp"


template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
struct fccn_basic {

    // Fully Connected Cascade Network.

    using fccn_base_ptr = fccn_basic*;
    using matrix = matrix<real, index, sfinae>;
    using matrix_ptr = matrix*;
    using reference = real&;
    using pointer = real*;
    using value_type = real;
    using idx_ptr = index*;

    fccn_basic ( index nrns_, const index iput_, const index oput_, const activation_function activation_function_, const real alpha_ );
    ~fccn_basic ( );

    index nrns, iput, oput, wgts, wgts_sqr, tr_patt, cols; // cols = tr_ibo->cols
    real alpha;

    real ( *activation ) ( const real, const real );

    pointer wts_data = nullptr;

    matrix_ptr tr_ibo = nullptr; pointer tr_ibo_data = nullptr;

    static index fccn_base_size ( ) noexcept { return sizeof ( fccn_basic ); }
    static fccn_base_ptr construct ( index nrns_, const index iput_, const index oput_, const activation_function activation_function_, const real alpha_ );

    void feedforward ( ) noexcept;

    static pointer convert ( const fmat *m_ );
};



template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
struct fccn;

// VNS.

template<typename real>
using vns_solution = std::vector<real>;

template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
struct vns_loss_function {
    using fccn_ptr = fccn<real, index, sfinae>*;
    fccn_ptr fp;
    real operator ( ) ( vns_solution<real> & solution_ ) const {
        return fp->feedforward_ase ( solution_.data ( ) );
    }
};

template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
using vns_local_search = hike::FILocalSearch<vns_solution<real>, vns_loss_function<real, index, sfinae>>;

template<typename real, typename index, typename sfinae = typename std::enable_if<are_valid_types<real, index>::value>::type>
using vns = hike::VNS<vns_solution<real>, vns_local_search<real, index, sfinae>>;


template<typename real, typename index, typename sfinae>
struct fccn {

    // Fully Connected Cascade Network.

    using fccn_ptr = fccn*;
    using matrix = matrix<real, index, sfinae>;
    using matrix_ptr = matrix*;
    using reference = real&;
    using pointer = real*;
    using pointer_ptr = pointer*;
    using value_type = real;
    using idx_ptr = index*;
    using generator = sax::Rng;

    index nrns, iput, oput, wgts, wgts_sqr, tr_patt, te_patt, cols, chro, prnt; // cols = tr_ibo->cols
    real alpha;

    real ( *activation ) ( const real, const real );
    real ( *activation_derivative ) ( const real, const real );

    pointer wts_data = nullptr; pointer wts_pdata = nullptr; // Previous version of weights.
    pointer jpm_data = nullptr; // j(acobian) for pattern M.
    idx_ptr jwc_data = nullptr; // jpm weights count (non-zero elements only) for respective tr_output neurons.
    idx_ptr nwc_data = nullptr;
    pointer gra_data = nullptr;
    pointer dlt_data = nullptr; pointer dlt_pdata = nullptr; // Previous version of deltas.
    pointer dia_data = nullptr;

    matrix_ptr tr_ibo = nullptr; pointer tr_ibo_data = nullptr;
    matrix_ptr te_ibo = nullptr; pointer te_ibo_data = nullptr;
    matrix_ptr tr_out = nullptr; pointer tr_out_data = nullptr;
    matrix_ptr te_out = nullptr; pointer te_out_data = nullptr;

    matrix_ptr nbn = nullptr; pointer nbn_data = nullptr;
    matrix_ptr hes = nullptr; pointer hes_data = nullptr;
    matrix_ptr ihs = nullptr; pointer ihs_data = nullptr; // Inverted Hessian.

    lapack_int *lapack_ipiv = nullptr;
    pointer lapack_work = nullptr;

    pointer pm_data = nullptr; pointer pm_front_tr_output = nullptr; // Holds copy of current pattern-line.
    pointer po_data = nullptr;

    pointer pop_data = nullptr; pointer_ptr srt_data = nullptr;

    fccn ( index nrns_, const index iput_, const index oput_, const activation_function activation_function_, const real alpha_ );
    ~fccn ( );

    static index no_wgts ( index nrns_, const index iput_, const index oput_ ) noexcept;
    static index fccn_size ( ) noexcept { return sizeof ( fccn ); }
    static fccn_ptr construct ( index nrns_, const index iput_, const index oput_, const activation_function activation_function_, const real alpha_ );

    template<typename T>
    void read ( T & col_data_, pointer f_, pointer i_, pointer o_, const index patt_, const index stride_, const activation_function activation_function_ ) noexcept;
    void read_in_out ( const std::string & file_name_, const activation_function activation_function_ );

    // GA.

    void construct_pop ( const index chro_, const index prnt_ );
    real mutate ( pointer c_, pointer p_ ) noexcept;
    void mutate2 ( pointer c_, pointer p_ ) noexcept;
    void crossover ( pointer c0_, pointer c1_, pointer p0_, pointer p1_ ) noexcept;
    real regenerate_pop ( ) noexcept;

    void activate ( pointer y_ ) noexcept;
    void feedforward ( ) noexcept;
    real feedforward_ase ( pointer w_ ) noexcept; // Same as above, returns average_squared_errors.

    pointer nguyen_widrow_weights ( pointer w_ ) noexcept;

    idx_ptr construct_jpm_wts_count ( ) const noexcept;
    idx_ptr construct_nrns_wts_count ( ) const noexcept;

    void fill_wgts_nbn ( ) noexcept;
    void fill_dlts_nbn ( ) noexcept;

    void construct_pat ( );
    void fill_pat ( const index pattern_ ) noexcept;

    real fill_hes_gra_pm_zero ( const index tr_output_neuron_ ) noexcept; // Helper to real fill_dia_hes_gra_ase ( ).
    real fill_hes_gra_pm ( const index tr_output_neuron_ ) noexcept; // Helper to real fill_dia_hes_gra_ase ( ).

    real fill_dia_hes_gra_ase ( ); // Returns average_squared_errors.

    lapack_int levenberg_marquardt_lu_invert_hes ( );
    lapack_int levenberg_marquardt_cholesky_invert_hes ( );
    lapack_int levenberg_marquardt_bunch_kaufman_invert_hes ( );
    lapack_int levenberg_marquardt_bunch_kaufman_rook_invert_hes ( );
    lapack_int levenberg_marquardt_aasen_invert_hes ( );

    void update_wts ( ) noexcept;
    real update_wts_asw ( ) noexcept; // Same as above, returns average_squared_weights.
    void reset_wts ( ) noexcept;

    void update_hes ( const real comb_coeff_ ) noexcept;

    real levenberg_marquardt_train1 ( index epochs_ = index { 10'000 } );
    real levenberg_marquardt_train2 ( );

    static void accumulate_average ( real & avg_, const real value_, const index cnt_ ) noexcept;
    static std::vector<index> to_gray ( index value_, index extended_digits_ ) noexcept;

    real average_squared_errors ( ) const noexcept;
    real average_squared_weights ( ) const noexcept;

    static pointer convert ( const fmat *m_ );
    void copy_wts ( const fmat *m_ );

    private:

    friend class cereal::access;

    template<class Archive>
    void save ( Archive & ar_ ) const;
    template<class Archive>
    void load ( Archive & ar_ );

    static auto seed_rng ( ) noexcept;
    static generator rng;
};

template<typename real, typename index, typename sfinae>
typename fccn<real, index, sfinae>::generator fccn<real, index, sfinae>::rng ( 123 ); // fccn<real, index, sfinae>::seed_rng ( ) );


#include "fccn.inl"
