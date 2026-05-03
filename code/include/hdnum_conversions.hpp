#pragma once

#include <cstddef>
#include "hdnum.hh"

namespace mpir {

// Vector conversion: resizes output if needed.
template<class T_out, class T_in>
void convert(hdnum::Vector<T_out>& out,
             const hdnum::Vector<T_in>& in)
{
    if (out.size() != in.size()) {
        out.resize(in.size());
    }

    for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = T_out(in[i]);
    }
}

// Convenience wrapper: allocates a new vector.
template<class T_out, class T_in>
hdnum::Vector<T_out> convert_vector(const hdnum::Vector<T_in>& in)
{
    hdnum::Vector<T_out> out(in.size());
    convert(out, in);
    return out;
}


// Matrix conversion: output matrix must already have correct shape.
template<class T_out, class T_in>
void convert(hdnum::DenseMatrix<T_out>& out,
             const hdnum::DenseMatrix<T_in>& in)
{
    if (out.rowsize() != in.rowsize() || out.colsize() != in.colsize()) {
        throw std::invalid_argument(
            "mpir::convert(DenseMatrix): output matrix has wrong size"
        );
    }

    for (std::size_t i = 0; i < in.rowsize(); ++i) {
        for (std::size_t j = 0; j < in.colsize(); ++j) {
            out[i][j] = T_out(in[i][j]);
        }
    }
}

// Convenience wrapper: allocates a new matrix.
template<class T_out, class T_in>
hdnum::DenseMatrix<T_out> convert_matrix(const hdnum::DenseMatrix<T_in>& in)
{
    hdnum::DenseMatrix<T_out> out(in.rowsize(), in.colsize());
    convert(out, in);
    return out;
}


// template<class T_out, class T_in>
// void convert(hdnum::DenseMatrix<T_out>& out,
//              const hdnum::DenseMatrix<T_in>& in)
// {
//     if (out.rowsize() != in.rowsize() || out.colsize() != in.colsize()) {
//         out.resize(in.rowsize(), in.colsize());
//     }

//     for (std::size_t i = 0; i < in.rowsize(); ++i) {
//         for (std::size_t j = 0; j < in.colsize(); ++j) {
//             out[i][j] = T_out(in[i][j]);
//         }
//     }
// }

// template<class T_out, class T_in>
// hdnum::DenseMatrix<T_out> convert_matrix(const hdnum::DenseMatrix<T_in>& in)
// {
//     hdnum::DenseMatrix<T_out> out(in.rowsize(), in.colsize());
//     convert(out, in);
//     return out;
// }


} // namespace mpir