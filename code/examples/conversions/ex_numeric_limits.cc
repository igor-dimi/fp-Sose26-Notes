#include <iostream>
#include <limits>

#include "hdnum.hh"

template<class T>
void print_numeric_limits_info(const char* name)
{
    std::cout << name << '\n';
    std::cout << "  is_specialized: "
              << std::numeric_limits<T>::is_specialized
              << '\n';

    if constexpr (std::numeric_limits<T>::is_specialized) {
        std::cout << "  epsilon: "
                  << std::numeric_limits<T>::epsilon()
                  << '\n';
    }

    std::cout << '\n';
}

int main()
{
    std::cout << std::boolalpha;

    print_numeric_limits_info<float>("float");
    print_numeric_limits_info<double>("double");

    print_numeric_limits_info<hdnum::FP32>("hdnum::FP32");
    print_numeric_limits_info<hdnum::FP64>("hdnum::FP64");

    print_numeric_limits_info<hdnum::FP16>("hdnum::FP16");
    print_numeric_limits_info<hdnum::bfloat16>("hdnum::bfloat16");

    print_numeric_limits_info<hdnum::FP128>("hdnum::FP128");
    print_numeric_limits_info<hdnum::FP256>("hdnum::FP256");
}