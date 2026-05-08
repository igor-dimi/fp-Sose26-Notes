#include <iostream>
#include "hdnum.hh"


int main(int argc, char const *argv[])
{
    std::cout << "HDNUM example\n";

    hdnum::Vector<double> x(3);
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    std::cout << "x = \n" << x << "\n";
    std::cout << "||x||_2 = \n" << x.two_norm() << "\n";
    

#ifdef HDNUM_HAS_GMP
    std::cout << "GMP support: enabled\n";

    hdnum::FP<128> a(1.0);
    hdnum::FP<128> b(3.0);
    hdnum::FP<128> c = a / b;

    std::cout << "FP<128> 1/3 = " << c << "\n";
#else
    std::cout << "GMP support: not enabled\n";
#endif


#ifdef HDNUM_HAS_CPFLOAT
    std::cout << "CPFloat support: enabled\n";

    hdnum::FP16 h = 1.0;
    hdnum::FP16 y = h / hdnum::FP16(3.0);

    std::cout << "FP16 1/3 = " << y << "\n";
#else
    std::cout << "CPFloat support: not enabled\n";
#endif


    return 0;
}

