#include <iostream>
#include <iomanip>
#include <string_view>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

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

template<class T_out, class T_in>
void test_scalar_cast(std::string_view out_name,
                      std::string_view in_name,
                      const T_in& x)
{
    T_out y = mpir::scalar_cast<T_out>(x);

    std::cout
        << " - scalar_cast<" << out_name << ">(" << in_name << "): "
        << std::setprecision(17)
        << approx_as_double(x)
        << " -> "
        << approx_as_double(y)
        << '\n';
}

#define TEST_SCALAR_CAST(T_out, T_in) \
    test_scalar_cast<T_out, T_in>(#T_out, #T_in, T_in(1.23456789012345))

int main()
{
    std::cout << std::boolalpha;

    std::cout << "double | float => double | float:\n";
    TEST_SCALAR_CAST(hdnum::FP64, hdnum::FP64);
    TEST_SCALAR_CAST(hdnum::FP64, hdnum::FP32);
    TEST_SCALAR_CAST(hdnum::FP32, hdnum::FP64);
    std::cout << '\n';

    std::cout << "CPFloat => CPFloat:\n";
    TEST_SCALAR_CAST(hdnum::FP16, hdnum::FP16);
    TEST_SCALAR_CAST(hdnum::FP16, hdnum::bfloat16);
    TEST_SCALAR_CAST(hdnum::bfloat16, hdnum::FP16);
    std::cout << '\n';

    std::cout << "FP => FP:\n";
    TEST_SCALAR_CAST(hdnum::FP256, hdnum::FP128);
    TEST_SCALAR_CAST(hdnum::FP256, hdnum::FP256);
    TEST_SCALAR_CAST(hdnum::FP128, hdnum::FP128);
    TEST_SCALAR_CAST(hdnum::FP128, hdnum::FP256);
    std::cout << '\n';

    std::cout << "double | float => CPFloat:\n";
    TEST_SCALAR_CAST(hdnum::FP16, hdnum::FP64);
    TEST_SCALAR_CAST(hdnum::bfloat16, hdnum::FP64);
    TEST_SCALAR_CAST(hdnum::FP16, hdnum::FP32);
    std::cout << '\n';

    std::cout << "CPFloat => double | float:\n";
    TEST_SCALAR_CAST(hdnum::FP64, hdnum::FP16);
    TEST_SCALAR_CAST(hdnum::FP64, hdnum::bfloat16);
    TEST_SCALAR_CAST(hdnum::FP32, hdnum::bfloat16);
    std::cout << '\n';

    std::cout << "CPFloat => FP:\n";
    TEST_SCALAR_CAST(hdnum::FP128, hdnum::FP16);
    TEST_SCALAR_CAST(hdnum::FP256, hdnum::FP16);
    TEST_SCALAR_CAST(hdnum::FP256, hdnum::bfloat16);
    std::cout << '\n';

    std::cout << "double | float => FP:\n";
    TEST_SCALAR_CAST(hdnum::FP256, hdnum::FP64);
    TEST_SCALAR_CAST(hdnum::FP128, hdnum::FP64);
    TEST_SCALAR_CAST(hdnum::FP128, hdnum::FP32);
    std::cout << '\n';

    std::cout << "FP => CPFloat:\n";
    TEST_SCALAR_CAST(hdnum::FP16, hdnum::FP128);
    TEST_SCALAR_CAST(hdnum::FP16, hdnum::FP256);
    TEST_SCALAR_CAST(hdnum::bfloat16, hdnum::FP128);
    TEST_SCALAR_CAST(hdnum::bfloat16, hdnum::FP256);
    std::cout << '\n';

    std::cout << "FP => double | float:\n";
    TEST_SCALAR_CAST(hdnum::FP64, hdnum::FP256);
    TEST_SCALAR_CAST(hdnum::FP64, hdnum::FP128);
    TEST_SCALAR_CAST(hdnum::FP32, hdnum::FP128);
    std::cout << '\n';

    return 0;
}