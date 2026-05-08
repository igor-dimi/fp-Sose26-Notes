#include <iostream>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

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
    
    hdnum::FP64 d = 4.0; // double
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

    return 0;
}
