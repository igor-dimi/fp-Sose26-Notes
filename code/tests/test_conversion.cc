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
    else
        return double(x);
}


// Floating point comparison helper
inline bool close(double a, double b, double atol = 0.0, double rtol = 1e-12)
{
    const double scale = std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= atol + rtol * scale;
}

inline bool close(double& a, double b, double atol = 0.0, double rtol = 1e-12)
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

    std::cout 
        << "scalar cast" << out_name << ">(" << in_name << "): "
        << x << " -> " << y << "\n";
}

template<class T_out, class T_in>
void test_scalar_cast(std::string_view out_name,
                      std::string_view in_name)
{
    std::cout << "\n[scalar]" << in_name << " -> " << out_name << '\n';

    test_scalar_cast_one_value<T_out>(T_in(0.0), out_name, in_name);
    test_scalar_cast_one_value<T_out>(T_in(1.0), out_name, in_name);
    test_scalar_cast_one_value<T_out>(T_in(-2.5), out_name, in_name);
    test_scalar_cast_one_value<T_out>(T_in(1.0 / 3.0), out_name, in_name);
    test_scalar_cast_one_value<T_out>(T_in(1.23456789012345), out_name, in_name);
}


// loss of precision
// fp64 -> cpfloat
template<class T_low>
void test_fp64_to_low_is_lossy(std::string_view low_name)
{
    std::cout << "\n[lossy conversion] FP64 -> " << low_name << '\n';
    const hdnum::FP64 x = 1.23456789012345;
    const T_low y = mpir::scalar_cast<T_low>(x);

    const auto y_double = approx_as_double<T_low>(y);

    std::cout << "original: " << x << '\n';
    std::cout << "lossy conversion: " << y << '\n';

    assert(y_double != x);
}

// Vector Tests


// function to fill vector of size 5 with values
template<class T_out, class T_in>
void fill_test_vector(hdnum::Vector<T_in>& v)
{
    assert(v.size() == 5);

    v[0] = T_in(0.0);
    v[1] = T_in(1.0);
    v[2] = T_in(-2.5);
    v[3] = T_in(1.0 / 3.0);
    v[4] = T_in(1.23456789012345);

}

template<class T_out, class T_in>
void test_vector_convert_size_zero(std::string_view out_name,
                                   std::string_view in_name)
{
    std::cout << "\n[vector size 0] " << in_name << " -> " << out_name << '\n';
    hdnum::Vector<T_in> in(0);
    hdnum::Vector<T_out> out(0);

    mpir::convert(out, in);

    assert(out.size() == 0);
}

template<class T_out, class T_in>
void test_vector_convert_size_one(std::string_view out_name, 
                                  std::string_view in_name)
{
    std::cout << "\n[vector size 1] " << in_name << " -> " << out_name << '\n';

    hdnum::Vector<T_in> in(1);
    in[0] = T_in(1.23456789012345);

    hdnum::Vector<T_out> out(1);
    mpir::convert(out, in);

    const T_out expected = mpir::scalar_cast<T_out>(in[0]);

    assert(close(approx_as_double(out[0]), approx_as_double(expected)));
}


template<class T_out, class T_in>
void test_vector_convert(std::string_view out_name,
                         std::string_view in_name)
{
    std::cout << "\n[vector] " << in_name << " -> " << out_name << '\n';

    hdnum::Vector<T_in> in(5);
    fill_test_vector<T_out>(in);

    hdnum::Vector<T_out> out(in.size());
    mpir::convert(out, in);

    for (std::size_t i = 0; i < in.size(); i++) {
        const T_out expected = mpir::scalar_cast<T_out>(in[i]);

        const double got_d = approx_as_double(out[i]);
        const double exp_d = approx_as_double(expected);

        assert(close(got_d, exp_d));
    }
}

template<class T_out, class T_in>
void test_vector_wrong_size_throws(std::string_view out_name,
                                   std::string_view in_name)
{
    std::cout << "\n[vector wrong size] " << in_name << " -> " << out_name << '\n';

    hdnum::Vector<T_in> in(5);
    hdnum::Vector<T_out> out(4);

    bool did_throw = false;

    try {
        mpir::convert(out, in);
    }
    catch (const std::invalid_argument&) {
        did_throw = true;
    }

    assert(did_throw);
}

template<class T_out, class T_in>
void test_convert_vector_convenience(std::string_view out_name,
                                     std::string_view in_name)
{
    std::cout << "\n[convert_vector convenicne] " 
              << in_name << " -> " << out_name << '\n';

    hdnum::Vector<T_in> in(5);
    fill_test_vector<T_out>(in);

    auto out = mpir::convert_vector<T_out>(in);

    static_assert(std::is_same_v<decltype(out), hdnum::Vector<T_out>>);

    assert(out.size() == in.size());

    for (std::size_t i = 0; i < in.size(); i++) {
        const T_out expected = mpir::scalar_cast<T_out>(in[i]);

        assert(close(
            approx_as_double(out[i]),
            approx_as_double(expected)
        ));
    }
}


// Matrix tests

template<class T_in>
void fill_test_matrix(hdnum::DenseMatrix<T_in>& A)
{
    for (std::size_t i = 0; i < A.rowsize(); ++i) {
        for (std::size_t j = 0; j < A.colsize(); ++i) {
            const double value = 
                0.25
                + static_cast<double>(i)
                - 0.125 * static_cast<double>(j)
                + 0.01 * static_cast<double>(i * j);

            A[i][j] = T_in(value);
        }
    }
}

template<class T_out, class T_in>
void test_matrix_convert_size_one(std::string_view out_name,
                                  std::string_view in_name)
{
    std::cout << "\n[matrix 1x1] " << in_name << " -> " << out_name << "\n";

    hdnum::DenseMatrix<T_in> in(1, 1);
    in[0][0] = T_in(1.23456789012345);

    hdnum::DenseMatrix<T_out> out(1, 1);
    mpir::convert(out, in);

    const T_out expected = mpir::scalar_cast<T_out>(in[0][0]);

    assert(close(
        approx_as_double(out[0][0]),
        approx_as_double(expected)
    ));
}

template<class T_out, class T_in>
void test_matrix_convert(std::string_view out_name,
                         std::string_view in_name)
{
    std::cout << "\n[matrix 5x5] " << in_name << " -> " << out_name << '\n';

    hdnum::DenseMatrix<T_in> in(5, 5);
    fill_test_matrix(in);

    hdnum::DenseMatrix<T_out> out(5, 5);
    mpir::convert(out, in);

    for(std::size_t i = 0; i < in.rowsize(); i++) {
        for (std::size_t j = 0; j < in.colsize(); j++) {
            const T_out expected = mpir::scalar_cast<T_out>(in[i][j]);

            assert(close(
                approx_as_double(out[i][j]),
                approx_as_double(expected)
            ));
        }
    }
}

template<class T_out, class T_in>
void test_matrix_wrong_size_throws(std::string_view out_name,
                                   std::string_view in_name)
{
    std::cout << "\n[matrix wrong size] " << in_name << " -> " << out_name << '\n';

    hdnum::DenseMatrix<T_in> in(5, 5);
    hdnum::DenseMatrix<T_out> out(4, 5);

    bool did_throw = false;

    try {
        mpir::convert(out, in);
    }
    catch(const std::invalid_argument&) {
        did_throw = true;
    }

    assert(did_throw);
}

template<class T_out, class T_in>
void test_convert_matrix_convenience(std::string_view out_name,
                                     std::string_view in_name)
{
    std::cout << "\n[convert_matrix convenience] "
              << in_name << " -> " << out_name << '\n';

    hdnum::DenseMatrix<T_in> in(5, 5);
    fill_test_matrix(in);

    auto out = mpir::convert_matrix<T_out>(in);

    static_assert(std::is_same_v<decltype(out), hdnum::DenseMatrix<T_out>>);

    assert(out.rowsize() == in.rowsize());
    assert(out.colsize() == in.colsize());

    for (std::size_t i = 0; i < in.rowsize(); i++) {
        for (std::size_t j = 0; j < in.colsize(); j++) {
            const T_out expected = mpir::scalar_cast<T_out>(in[i][j]);

            // auto t = approx_as_double(out[i][j]);

            // assert(close(
            //     approx_as_double(t,
            //     approx_as_double(expected))
            // ));
        }
    }
}

template<class T_out, class T_in>
void test_all_for_pair(std::string_view out_name,
                       std::string_view in_name)
{
    test_scalar_cast<T_out, T_in>(out_name, in_name);

    test_vector_convert<T_out, T_in>(out_name, in_name);
    test_vector_convert_size_one<T_out, T_in>(out_name, in_name);
    test_vector_convert_size_zero<T_out, T_in>(out_name, in_name);
    test_vector_wrong_size_throws<T_out, T_in>(out_name, in_name);
    test_convert_vector_convenience<T_out, T_in>(out_name, in_name);

    test_matrix_convert<T_out, T_in>(out_name, in_name);
    test_matrix_convert_size_one<T_out, T_in>(out_name, in_name);
    test_matrix_wrong_size_throws<T_out, T_in>(out_name, in_name);
    test_convert_matrix_convenience<T_out, T_in>(out_name, in_name);
}


} // namespace test


#define TEST_PAIR(T_out, T_in) \
    test::test_all_for_pair<T_out, T_in>(#T_out, #T_in)


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

    // Native <-> native
    TEST_PAIR(hdnum::FP64, hdnum::FP32);
    TEST_PAIR(hdnum::FP32, hdnum::FP64);


    // Working preision u -> low precision u_f
    TEST_PAIR(hdnum::FP16, hdnum::FP64);
    TEST_PAIR(hdnum::bfloat16, hdnum::FP64);
    TEST_PAIR(hdnum::FP8, hdnum::FP64);

    // Low precision u_f -> working precision u
    TEST_PAIR(hdnum::FP64, hdnum::FP16);
    TEST_PAIR(hdnum::FP64, hdnum::bfloat16);
    TEST_PAIR(hdnum::FP64, hdnum::FP8);


    // Working precision u -> residual precision u_r
    TEST_PAIR(hdnum::FP128, hdnum::FP64);
    TEST_PAIR(hdnum::FP256, hdnum::FP64);


    // Residual precision u_r -> working precision u
    TEST_PAIR(hdnum::FP64, hdnum::FP128);
    TEST_PAIR(hdnum::FP64, hdnum::FP256);


    // Low precision u_f <-> high residual / reference precision u_r
    // not strictly necessary for the algorithm
    TEST_PAIR(hdnum::FP128, hdnum::FP16);
    TEST_PAIR(hdnum::FP16, hdnum::FP128);
    TEST_PAIR(hdnum::FP256, hdnum::bfloat16);
    TEST_PAIR(hdnum::bfloat16, hdnum::FP256);


    // High <-> high
    TEST_PAIR(hdnum::FP256, hdnum::FP128);
    TEST_PAIR(hdnum::FP128, hdnum::FP256);


    // Explicit lossy / non-lossy sanity checks
    test::test_fp64_to_low_is_lossy<hdnum::FP16>("hdnum::FP16");
    test::test_fp64_to_low_is_lossy<hdnum::bfloat16>("hdnum::bfloat16");
    test::test_fp64_to_low_is_lossy<hdnum::FP8>("hdnum::FP8");


    return 0;
}