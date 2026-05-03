#include <iostream>
#include <cassert>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

int main()
{
    {
        hdnum::Vector<double> x(3);
        x[0] = 1.0;
        x[1] = 2.0;
        x[2] = 3.0;

        std::cout << "x = \n" << x << std::endl;

        hdnum::Vector<hdnum::FP<128>> y;
        mpir::convert(y, x);

        assert(y.size() == x.size());

        std::cout << "Vector conversion double -> FP<128>:\n";
        std::cout << y << "\n";
    }

#ifdef HDNUM_HAS_CPFLOAT
    {
        hdnum::Vector<double> x(3);
        x[0] = 1.0;
        x[1] = 1.0 / 3.0;
        x[2] = 1000.0;

        auto y = mpir::convert_vector<hdnum::FP16>(x);
        auto z = mpir::convert_vector<double>(y);

        std::cout << "Vector conversion double -> FP16 -> double:\n";
        std::cout << z << "\n";
    }
#endif

    {
        hdnum::DenseMatrix<double> A(2, 2);
        A[0][0] = 1.0;
        A[0][1] = 2.0;
        A[1][0] = 3.0;
        A[1][1] = 4.0;

        std::cout << "A = \n" << A << std::endl;

        hdnum::DenseMatrix<hdnum::FP<128>> B(2, 2);
        mpir::convert(B, A);

        assert(B.rowsize() == A.rowsize());
        assert(B.colsize() == A.colsize());

        std::cout << "Matrix conversion double -> FP<128>:\n";
        std::cout << "B = \n" << B << "\n";
    }

    std::cout << "All conversion tests passed.\n";
    return 0;
}