#include "hdnum.hh"
#include "hdnum_conversions.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
    hdnum::Vector<hdnum::FP128> vec(10);
    std::cout << vec << "\n";
    auto n = norm(vec);
    std::cout << "norm: " << n << "\n";
    for (int i = 0; i < vec.size(); i++) vec[i] = mpir::scalar_cast<hdnum::FP128>(i);
    std::cout << vec << "\n";
    n = norm(vec);
    std::cout << "norm: " << n << "\n";
    return 0;
}
