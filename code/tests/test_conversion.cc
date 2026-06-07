#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"


namespace test {

template<class T>
double approx_as_double(const T& x)
{
    if constexpr (mpir::is_hdnum_fp_v<T>) {
        return x.getNumber().get_d();
    }
    else {
        return static_cast<double>(x);
    }
}

// Floating point comparison helper
inline bool close(double a, double b, double atol = 0.0, double rtol = 1e-12)
{
    const double scale = std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= atol + rtol * scale;
}

// test scalar_cast on one value
template <class T_out, class T_in>
void test_scalar_cast_one_value(const T_in& x,
        std::string_view out_name,
        std::string_view in_name)
{
    auto y = mpir::scalar_cast<T_out>(x);

    // compile-time check: scalar_cast<T_out>(...) really returns T_out
    static_assert(std::is_same_v<decltype(y), T_out>);

    const double xd = approx_as_double(x);
    const double yd = approx_as_double(y);

    std::cout 
        << "scalar_cast<" << out_name << ">(" << in_name << "): "
        << xd << " -> " << yd << '\n';
}

template<class T_out, class T_in>
void test_scalar_cast(std::string_view out_name,
                      std::string_view in_name)
{
    std::cout << "\n[scalar] " << in_name << " -> " << out_name << '\n';

    test_scalar_cast_one_value<T_out>(T_in(0.0), out_name, in_name);
    test_scalar_cast_one_value<T_out>(T_in(1.0), out_name, in_name);
    test_scalar_cast_one_value<T_out>(T_in(-2.5), out_name, in_name);
    test_scalar_cast_one_value<T_out>(T_in(1.0 / 3.0), out_name, in_name);
    test_scalar_cast_one_value<T_out>(T_in(1.23456789012345), out_name, in_name);
}

// lossy conversion tests
template<class T_low>
void test_fp64_to_low_is_lossy(std::string_view low_name)
{
    std::cout << "\n[lossy conversion] hdnum::FP64 -> " << low_name << "\n";

    const hdnum::FP64 x = 1.23456789012345;

    const T_low y = mpir::scalar_cast<T_low>(x);

    const double xd = approx_as_double(x);
    const double yd = approx_as_double(y);

    std::cout << "original: "  << xd << '\n';
    std::cout << "lossy-conversion: "  << yd << '\n';

    assert(xd != yd);
}

// lossless conversion tets
template<class T_high, class T_low>
void test_low_to_high_is_lossless(
    std::string_view high_name, 
    std::string_view low_name
)
{
    std::cout << "\n[non-lossy conversion] " << low_name << " -> "  << high_name << '\n';
    const T_low x = T_low(1.2345678);
    auto y = mpir::scalar_cast<T_high>(x);
    
    auto xd = approx_as_double(x);
    auto yd = approx_as_double(y);

    std::cout << "low original: " << xd << '\n';
    std::cout << "conversion to high: " << yd << '\n';

    assert(close(xd, yd));

}

// vector tests

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
void test_vector_convert(std::string_view out_name,
                         std::string_view in_name)
{
    std::cout << "\n[vector] " << in_name << " -> "  << out_name << '\n';

    hdnum::Vector<T_in> in(5);
    fill_test_vector<T_out>(in);

    hdnum::Vector<T_out> out(in.size());
    mpir::convert(out, in);


    std::cout << "in: " << in << '\n' << "out: " << out << '\n';

    assert(out.size() == in.size());

    for (std::size_t i = 0; i < in.size(); ++i) {
        const T_out expected = mpir::scalar_cast<T_out>(in[i]);

        const double got_d = approx_as_double(out[i]);
        const double exp_d = approx_as_double(expected);

        assert(close(got_d, exp_d));
    }
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

    std::cout << "in: " << in << '\n' << "out: " << out << '\n';

    const T_out expected = mpir::scalar_cast<T_out>(in[0]);
    assert(close(approx_as_double(out[0]), approx_as_double(expected)));
}

template<class T_out, class T_in>
void test_vector_convert_size_zero(std::string_view out_name,
                                   std::string_view in_name)
{
    std::cout << "\n[vector size 0] " << in_name << " -> " << out_name << '\n';

    hdnum::Vector<T_in> in(0);
    hdnum::Vector<T_out> out(0);

    mpir::convert(out, in);

    std::cout << "in: " << in << '\n' << "out: " << out << '\n';

    assert(out.size() == 0);
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
    if (did_throw) std::cout << "error thrown: " << did_throw << '\n';
    assert(did_throw);
}


template<class T_out, class T_in>
void test_convert_vector_convenience(std::string_view out_name,
    std::string_view in_name)
{
    std::cout << "\n[convert_vector convenience function] "
              << in_name << " -> " << out_name << '\n';

    hdnum::Vector<T_in> in(5);
    fill_test_vector<T_out>(in);

    auto out = mpir::convert_vector<T_out>(in);

    static_assert(std::is_same_v<decltype(out), hdnum::Vector<T_out>>);

    assert(out.size() == in.size());

    std::cout << "in: " << in << "out: " << out << '\n';

    for (std::size_t i = 0; i < in.size(); i++) {
        const T_out expected = mpir::scalar_cast<T_out>(in[i]);

        assert(close(
            approx_as_double(out[i]),
            approx_as_double(expected)
        ));
    }
                
}


// matrix tests:
template<class T_in>
void fill_test_matrix(hdnum::DenseMatrix<T_in>& A)
{
    for (std::size_t i = 0; i < A.rowsize(); i++) {
        for (std::size_t j = 0; j < A.colsize(); j++) {
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
void test_matrix_convert(std::string_view out_name,
                         std::string_view in_name)
{
    std::cout << "\n[matrix 5x5] " << in_name << " -> " << out_name << '\n';

    hdnum::DenseMatrix<T_in> in(5, 5);
    fill_test_matrix(in);

    hdnum::DenseMatrix<T_out> out(5, 5);
    mpir::convert(out, in);


    std::cout << "in: " << in << "out: " << out << '\n';

    for (std::size_t i = 0; i < in.rowsize(); ++i){
        for (std::size_t j = 0; j < in.colsize(); ++j) {
            const T_out expected = mpir::scalar_cast<T_out>(in[i][j]);

            assert(close(
                approx_as_double(out[i][j]),
                approx_as_double(expected)   
            ));
        }
    }
}

template<class T_out, class T_in>
void test_matrix_convert_size_one(std::string_view out_name,
                                  std::string_view in_name)
{
    std::cout << "\n[matrix 1x1] " << in_name << " -> " << out_name << '\n';
    
    hdnum::DenseMatrix<T_in> in(1, 1);
    in[0][0] = T_in(1.23456789012345);

    hdnum::DenseMatrix<T_out> out(1, 1);
    mpir::convert(out, in);

    std::cout << "in: " << in << "out: " << out << '\n';

    const T_out expected = mpir::scalar_cast<T_out>(in[0][0]);

    assert(close(
        approx_as_double(out[0][0]),
        approx_as_double(expected)
    ));
}

template<class T_out, class T_in>
void test_matrix_wrong_size_throws(std::string_view out_name,
                                   std::string_view in_name)
{
    std::cout << "\n[matrix wrong size throws] " << in_name << " -> " << out_name << '\n';

    hdnum::DenseMatrix<T_in> in(5, 5);
    hdnum::DenseMatrix<T_out> out(4, 5);

    std::cout << "in: " << in << "out: " << out << '\n';

    bool did_throw = false;

    try {
        mpir::convert(out, in);
    }
    catch (const std::invalid_argument&) {
        did_throw = true;
    }

    assert(did_throw);

    std::cout << "did throw: " << did_throw << '\n';

}

template<class T_out, class T_in>
void test_convert_matrix_convenience(
    std::string_view out_name,
    std::string_view in_name
)
{
    std::cout << "\n[convert_matrix convenience function] "
              << in_name << " -> " << out_name << '\n';

    hdnum::DenseMatrix<T_in> in(5, 5);
    fill_test_matrix(in);

    auto out = mpir::convert_matrix<T_out>(in);

    std::cout << "in: " << in << "out: " << out << '\n';

    static_assert(std::is_same_v<decltype(out), hdnum::DenseMatrix<T_out>>);

    for (std::size_t i = 0; i < in.rowsize(); ++i) {
        for (std::size_t j = 0; j < in.colsize(); ++j) {
            const T_out expected = mpir::scalar_cast<T_out>(in[i][j]);

            assert(close(
                approx_as_double(out[i][j]),
                approx_as_double(expected)
            ));
        }
    }


}

template<class T_out, class T_in>
void test_all_for_pair(
    std::string_view out_name,
    std::string_view in_name
)
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
    std::cout << std::boolalpha;
    std::cout << std::setprecision(17);
    auto y = hdnum::FP128(1.23456789);
    auto x = test::approx_as_double(y);
    std::cout << x << '\n';
    std::cout << y << '\n';


    //scalar tests
    test::test_scalar_cast_one_value<hdnum::FP128, double>(1.23456789012345, "hdnum::FP128", "double");
    test::test_scalar_cast<hdnum::bfloat16, hdnum::FP128>("bfloat16", "hdnum::FP128");
    test::test_fp64_to_low_is_lossy<hdnum::FP16>("hdnum::FP16");
    test::test_low_to_high_is_lossless<hdnum::FP128, hdnum::FP16>(
        "hdnum::FP128", "hdnum::FP16"
    );
    test::test_low_to_high_is_lossless<hdnum::FP128, hdnum::bfloat16>(
        "hdnum::FP128", "hdnum::bfloat16"
    );
    test::test_low_to_high_is_lossless<hdnum::FP256, hdnum::FP16>(
        "hdnum::FP256", "hdnum::FP16"
    );

    // vector tests
    test::test_vector_convert<hdnum::FP16, hdnum::FP128>("hdnum::FP16", "hdnum::FP128");
    test::test_vector_convert_size_one<hdnum::FP16, hdnum::FP128>("hdnum::FP16", "hdnum::FP128");
    test::test_vector_convert_size_zero<hdnum::FP16, hdnum::FP128>("hdnum::FP16", "hdnum::FP128");
    test::test_vector_wrong_size_throws<hdnum::FP16, hdnum::FP128>("hdnum::FP16", "hdnum::FP128");
    test::test_convert_vector_convenience<hdnum::FP16, hdnum::FP128>("hdnum::FP16", "hdnum::FP128");

    // matrix tests
    hdnum::DenseMatrix<hdnum::FP256> M(5, 5);
    std::cout << "fp256 Matrix M: " << M << "\n";
    test::fill_test_matrix(M);
    std::cout << "fp256 Matrix M after fill: " << M << "\n";
    test::test_matrix_convert<hdnum::FP8, hdnum::FP128>("hdnum::FP8", "hdnum::FP128");
    test::test_matrix_convert_size_one<hdnum::FP8, hdnum::FP128>("hdnum::FP8", "FP::FP128");
    test::test_matrix_wrong_size_throws<hdnum::FP16, hdnum::FP128>(
        "hdnum::FP16", "hdnum::FP128"
    );
    test::test_convert_matrix_convenience<hdnum::FP8, hdnum::FP128>(
        "hdnum::FP16", "hdnum::FP128"
    );


    // Native <-> native
    TEST_PAIR(hdnum::FP64, hdnum::FP32);
    TEST_PAIR(hdnum::FP32, hdnum::FP64);

    // Working precision u -> low precision uf
    TEST_PAIR(hdnum::FP16, hdnum::FP64);
    TEST_PAIR(hdnum::bfloat16, hdnum::FP64);
    TEST_PAIR(hdnum::FP8, hdnum::FP64);

    // Low precision uf -> working precision u
    TEST_PAIR(hdnum::FP64, hdnum::FP16);
    TEST_PAIR(hdnum::FP64, hdnum::bfloat16);
    TEST_PAIR(hdnum::FP64, hdnum::FP8);

    // Working precision u -> residual precision ur
    TEST_PAIR(hdnum::FP128, hdnum::FP64);
    TEST_PAIR(hdnum::FP256, hdnum::FP64);

    // Residual precision ur -> working precision u
    TEST_PAIR(hdnum::FP64, hdnum::FP128);
    TEST_PAIR(hdnum::FP64, hdnum::FP256);

    // Low precision uf <-> high residual/reference precision ur
    TEST_PAIR(hdnum::FP128, hdnum::FP16);
    TEST_PAIR(hdnum::FP16, hdnum::FP128);
    TEST_PAIR(hdnum::FP256, hdnum::bfloat16);
    TEST_PAIR(hdnum::bfloat16, hdnum::FP256);

    // High <-> high
    TEST_PAIR(hdnum::FP256, hdnum::FP128);
    TEST_PAIR(hdnum::FP128, hdnum::FP256);

    // -------------------------------------------------------------------------
    // Explicit lossy 
    // -------------------------------------------------------------------------
    test::test_fp64_to_low_is_lossy<hdnum::FP16>("hdnum::FP16");
    test::test_fp64_to_low_is_lossy<hdnum::bfloat16>("hdnum::bfloat16");
    test::test_fp64_to_low_is_lossy<hdnum::FP8>("hdnum::FP8");

    // Explicity lossess conversion
    test::test_low_to_high_is_lossless<hdnum::FP128, hdnum::FP16>(
        "hdnum::FP128", "hdnum::FP16"
    );
    test::test_low_to_high_is_lossless<hdnum::FP128, hdnum::bfloat16>(
        "hdnum::FP128", "hdnum::bfloat16"
    );
    test::test_low_to_high_is_lossless<hdnum::FP256, hdnum::FP16>(
        "hdnum::FP256", "hdnum::FP16"
    );


    std::cout << "\nAll conversion tests passed.\n";

    return 0;

}
