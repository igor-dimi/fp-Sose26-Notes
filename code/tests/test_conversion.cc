#include <iostream>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

namespace test {

// common scalar inspection helper
template<class T>
double approx_as_double(const T& x)
{
    if constexpr (mpir::is_hdnum_fp_v<T>)
        return x.getNumber().get_d();
    return static_cast<double>(x);
}


// Floating point comparison helper
inline bool close(double a, double b, double atol = 0.0, double rtol = 1e-12)
{
    const double scale = std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= atol + rtol * scale;
}

template<class T_out, class T_in>
void test_scalar_cast_one_value(const T_in& x,
                                std::string_view out_name,
                                std::string_view in_name)
{
    auto y = mpir::scalar_cast<T_out>(x);

    // compile-time check: scalar_cast<T_out>(...) really returns T_out
    static_assert(std::is_same_v<decltype(y), T_out>);

    const double xd = approx_as_double(x);
    const double yd = approx_as_double(y);
}

}

int main()
{
//     {
//         hdnum::Vector<double> x(3);
//         x[0] = 1.0;
//         x[1] = 2.0;
//         x[2] = 3.0;

//         std::cout << "x = \n" << x << std::endl;

//         hdnum::Vector<hdnum::FP<128>> y;
//         mpir::convert(y, x);

//         assert(y.size() == x.size());

//         std::cout << "Vector conversion double -> FP<128>:\n";
//         std::cout << y << "\n";
//     }

// #ifdef HDNUM_HAS_CPFLOAT
//     {
//         hdnum::Vector<double> x(3);
//         x[0] = 1.0;
//         x[1] = 1.0 / 3.0;
//         x[2] = 1000.0;

//         auto y = mpir::convert_vector<hdnum::FP16>(x);
//         auto z = mpir::convert_vector<double>(y);

//         std::cout << "Vector conversion double -> FP16 -> double:\n";
//         std::cout << z << "\n";
//     }
// #endif

//     {
//         hdnum::DenseMatrix<double> A(2, 2);
//         A[0][0] = 1.0;
//         A[0][1] = 2.0;
//         A[1][0] = 3.0;
//         A[1][1] = 4.0;

//         std::cout << "A = \n" << A << std::endl;

//         hdnum::DenseMatrix<hdnum::FP128> B(2, 2);
//         mpir::convert(B, A);

//         assert(B.rowsize() == A.rowsize());
//         assert(B.colsize() == A.colsize());

//         std::cout << "Matrix conversion double -> FP<128>:\n";
//         std::cout << "B = \n" << B << "\n";
//     }

//     std::cout << "All conversion tests passed.\n";





    return 0;
}