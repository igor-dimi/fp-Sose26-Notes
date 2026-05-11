#include <iostream>
#include <type_traits>

#include "hdnum.hh"

template<class T_out, class T_in>
void print_constructible(std::string_view out_name,
                         std::string_view in_name)
{
    std::cout
        << " - is " << out_name
        << " constructible from const " << in_name << "&?: "
        << std::is_constructible_v<T_out, const T_in&>
        << '\n';
}

#define PRINT_CONSTRUCTIBLE(T_out, T_in) \
    print_constructible<T_out, T_in>(#T_out, #T_in)


int main()
{

    std::cout << std::boolalpha;  
    
    // basic floating point types, float and double
    std::cout << "double | float => double | float:" << std::endl;
    std::cout << " - is hdnum::FP64 constructible from hdnum::FP64?: " << std::is_constructible_v<hdnum::FP64, const hdnum::FP64&> << std::endl;
    std::cout << " - is hdnum::FP64 constructible from hdnum::FP32?: " << std::is_constructible_v<hdnum::FP64, const hdnum::FP32&> << std::endl;
    std::cout << "\n";

    // CPFloat => CPFloat
    std::cout << "CPFloat => CPFloat:" << "\n";
    std::cout << " - is FP16 constructible from FP16?: " << std::is_constructible_v<hdnum::FP16, hdnum::FP16> << std::endl;
    std::cout << " - is FP16 constructible from bfloat16?: " << std::is_constructible_v<hdnum::FP16, hdnum::bfloat16> << std::endl;
    std::cout << " - is bfloat16 constructible from FP16?:  " <<  std::is_constructible_v<hdnum::bfloat16, hdnum::FP16> << std::endl;
    std::cout << "\n";

    // FP => FP
    std::cout << "FP => FP:\n";
    std::cout << " - is FP256 constructible from FP128?: " << std::is_constructible_v<hdnum::FP256, hdnum::FP128> << std::endl;
    PRINT_CONSTRUCTIBLE(hdnum::FP256, hdnum::FP256);
    PRINT_CONSTRUCTIBLE(hdnum::FP128, hdnum::FP128);
    PRINT_CONSTRUCTIBLE(hdnum::FP128, hdnum::FP256);
    std::cout << "\n";

    // double | float => CPFloat
    std::cout << "double | float => CPFloat:\n";
    PRINT_CONSTRUCTIBLE(hdnum::FP16, hdnum::FP64);
    PRINT_CONSTRUCTIBLE(hdnum::bfloat16, hdnum::FP64);
    PRINT_CONSTRUCTIBLE(hdnum::FP16, hdnum::FP32);
    std::cout << "\n";

    // CPFloat => double
    std::cout << "CPFloat => double | float:\n";
    PRINT_CONSTRUCTIBLE(hdnum::FP64, hdnum::FP16);
    PRINT_CONSTRUCTIBLE(hdnum::FP64, hdnum::bfloat16);
    PRINT_CONSTRUCTIBLE(hdnum::FP32, hdnum::bfloat16);
    std::cout << "\n";

    // CPFloat => FP
    std::cout << "CPFloat => FP:\n";
    PRINT_CONSTRUCTIBLE(hdnum::FP128, hdnum::FP16);
    PRINT_CONSTRUCTIBLE(hdnum::FP256, hdnum::FP16);
    PRINT_CONSTRUCTIBLE(hdnum::FP256, hdnum::bfloat16);
    std::cout << "\n";

    // double | float => FP
    std::cout << "double | float => FP:\n";
    PRINT_CONSTRUCTIBLE(hdnum::FP256, hdnum::FP64);
    PRINT_CONSTRUCTIBLE(hdnum::FP128, hdnum::FP64);
    PRINT_CONSTRUCTIBLE(hdnum::FP128, hdnum::FP32);
    std::cout << "\n";

    // FP => CPFloat
    std::cout << "FP => CPFloat:\n";
    PRINT_CONSTRUCTIBLE(hdnum::FP16, hdnum::FP128);
    PRINT_CONSTRUCTIBLE(hdnum::FP16, hdnum::FP256);
    PRINT_CONSTRUCTIBLE(hdnum::bfloat16, hdnum::FP128);
    PRINT_CONSTRUCTIBLE(hdnum::bfloat16, hdnum::FP256);
    std::cout << "\n";

    // FP => double | float
    std::cout << "FP => double | float:\n";
    PRINT_CONSTRUCTIBLE(hdnum::FP64, hdnum::FP256);
    PRINT_CONSTRUCTIBLE(hdnum::FP64, hdnum::FP128);
    PRINT_CONSTRUCTIBLE(hdnum::FP32, hdnum::FP128);
    std::cout << "\n";

    return 0;
}
