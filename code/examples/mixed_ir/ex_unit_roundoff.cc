#include <iostream>
#include <iomanip>

#include "hdnum.hh"
#include "unit_roundoff.hpp"

int main()
{
    std::cout << std::scientific << std::setprecision(17);

    std::cout << "Unit roundoffs used by mpir::default_unit_roundoff<T>()\n\n";

    std::cout << "FP32   : " << mpir::default_unit_roundoff<hdnum::FP32>()
              << "   expected 2^-24\n";

    std::cout << "FP64   : " << mpir::default_unit_roundoff<hdnum::FP64>()
              << "   expected 2^-53\n";

#ifdef HDNUM_HAS_CPFLOAT
    std::cout << "FP8    : " << mpir::default_unit_roundoff<hdnum::FP8>()
              << "   expected 2^-4\n";

    std::cout << "bfloat16: " << mpir::default_unit_roundoff<hdnum::bfloat16>()
              << "   expected 2^-8\n";

    std::cout << "FP16   : " << mpir::default_unit_roundoff<hdnum::FP16>()
              << "   expected 2^-11\n";
#endif

#ifdef HDNUM_HAS_GMP
    std::cout << "FP128  : " << mpir::default_unit_roundoff<hdnum::FP128>()
              << "   expected 2^-64\n";

    std::cout << "FP256  : " << mpir::default_unit_roundoff<hdnum::FP256>()
              << "   expected 2^-192\n";

    std::cout << "FP512  : " << mpir::default_unit_roundoff<hdnum::FP512>()
              << "   expected 2^-448\n";

    std::cout << "FP1024 : " << mpir::default_unit_roundoff<hdnum::FP1024>()
              << "   expected 2^-960\n";
#endif

    return 0;
}