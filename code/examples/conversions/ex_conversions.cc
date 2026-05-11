#include <iostream>
#include <limits>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

#include <type_traits>

#define SHOW(x) std::cout << #x << " = " << (x) << '\n'

template <typename... Ts>
void print_line(const Ts&... values)
{
    (std::cout << ... << values) << '\n';
}

template <typename... Ts>
void print_all(const Ts&... values)
{
    ((std::cout << values << '\n'), ...);
}

int main()
{
    std::cout << std::setprecision(50);
    
    hdnum::FP64 d = 3.14159265359; // double
    // hdnum::FP32 f = 4.0; // float


    // double => CPFloat
    hdnum::bfloat16 bf(d); 
    hdnum::FP16 fp16(d);
    hdnum::FP8 fp8(d);

    // double => GMP
    hdnum::FP128 fp128(d);
    hdnum::FP256 fp256(d);

    // CPFloat => double works because CPFLoat has double() operator
    hdnum::FP64 bfl16_to_double(bf);
    hdnum::FP64 fp16_to_double(fp16);
    hdnum::FP64 fp8_to_double(fp8);

    // naive GMP => double doesn't work because GMP doesn't have double() operator
    // hdnum::FP64 fp128_to_double(fp128);
    // hdnum::FP64 fp256_to_double(fp256);
    // instead
    hdnum::FP64 fp128_to_double = fp128.getNumber().get_d();
    hdnum::FP64 fp256_to_double = fp256.getNumber().get_d();

    // CPFLoat => GMP works because  CPFloat is implicityly converted to double,via double()
    hdnum::FP128 fp16_to_fp128(fp16);
    hdnum::FP256 fp16_to_fp256(fp16);


    // naive GMP => CPFloat doesn't work because CPFloat doesn't have appropriate constructor
    // hdnum::FP16 fp128_to_fp16(fp128);


    std::cout << "PRINT\n\n";
    SHOW(d);
    SHOW(bf);
    SHOW(fp16);
    SHOW(fp8);
    SHOW(bfl16_to_double);
    SHOW(fp16_to_double);
    SHOW(fp8_to_double);
    SHOW(fp128);
    SHOW(fp256);
    SHOW(fp16_to_fp128);

    // print_all(
    //     d,
    //     bf,
    //     fp16,
    //     fp8,
    //     bfl16_to_double,
    //     fp16_to_double,
    //     fp8_to_double,
    //     fp128,
    //     fp256,
    //     fp16_to_fp128
    // );
    std::cout << "\nEND\n";


    auto d2 = fp128.getNumber().get_d();

    std::cout << d2 << std::endl;

    double x = 1.13315;
    float x2 = x;
    auto fp_low = hdnum::FP16(x);
    auto fp_low2 = hdnum::bfloat16(x);

    SHOW(fp_low);
    SHOW(fp_low2);
    SHOW(x2);
    SHOW(x);

    auto x3 = double(fp_low);

    SHOW(x3);

    hdnum::FP256 fp256_number = 1.234567;
    auto d4 = fp256_number.getNumber().get_d();


    SHOW(fp256_number);
    SHOW(d4);

    mpf_class m("1.123456789012345678901234567890", 128);
    hdnum::FP128 x001(m);


    SHOW(x001);
    {
        hdnum::FP16 x = 1.0;
        hdnum::FP256 z = 1.34;
        double y = x;
        auto y2 = double(x);
        auto y3 = float(x);
        SHOW(x);
        SHOW(y);
        SHOW(y2);
        SHOW(y3);
        std::cout << sizeof(y3) << std::endl;
        auto z3 = static_cast<hdnum::FP256>(x);
        SHOW(z3);
        std::cout << "size of z: " << sizeof(z3) << std::endl;
    }

    std::cout << std::is_constructible_v<double, double> << std::endl;
    std::cout << std::is_constructible_v<hdnum::FP128, hdnum::FP16> << std::endl;
    std::cout << "is FP64 constructible from FP16?:  " <<  std::is_constructible_v<hdnum::FP64, hdnum::FP16> << std::endl;
    std::cout << std::is_constructible_v<hdnum::FP64, hdnum::FP16> << std::endl;


    hdnum::FP256 x128 = 1.145;
    std::cout << "new new " << std::endl;
    auto y128 = mpir::scalar_cast<hdnum::FP16, hdnum::FP256>(x128);

    SHOW(y128);
    std::cout << "size of y128: " << sizeof(y128) << std::endl;

    return 0;
}
