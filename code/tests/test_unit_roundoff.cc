#include <cmath>
#include <iostream>
#include <iomanip>

#include "hdnum.hh"
#include "unit_roundoff.hpp"

void check(const char* name, double got, int p)
{
    const double expected = std::ldexp(1.0, -p); // exactly 2^-p, if representable
    const double diff = std::abs(got - expected);

    std::cout << std::left << std::setw(12) << name
              << " got = " << std::scientific << std::setprecision(17) << got
              << " expected = " << expected
              << " diff = " << diff
              << '\n';
}

int main()
{
    check("FP32",   mpir::default_unit_roundoff<hdnum::FP32>(),   24);
    check("FP64",   mpir::default_unit_roundoff<hdnum::FP64>(),   53);

#ifdef HDNUM_HAS_CPFLOAT
    check("FP8",     mpir::default_unit_roundoff<hdnum::FP8>(),       4);
    check("bfloat16", mpir::default_unit_roundoff<hdnum::bfloat16>(), 8);
    check("FP16",    mpir::default_unit_roundoff<hdnum::FP16>(),     11);
#endif

#ifdef HDNUM_HAS_GMP
    check("FP128",  mpir::default_unit_roundoff<hdnum::FP128>(),  64);
    check("FP256",  mpir::default_unit_roundoff<hdnum::FP256>(),  192);
    check("FP512",  mpir::default_unit_roundoff<hdnum::FP512>(),  448);
    check("FP1024", mpir::default_unit_roundoff<hdnum::FP1024>(), 960);
#endif
}