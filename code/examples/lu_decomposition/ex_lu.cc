#include <iostream>
#include "hdnum.hh"

#define PRINT(x) \
    do { std::cout << #x << ": " << (x) << std::endl; } while (false)

int main(int argc, const char** argv) {

    hdnum::Vector<hdnum::FP256> b(3);
    b[0] = 15;
    b[1] = 73;
    b[2] = 12;

    hdnum::DenseMatrix <hdnum::FP256> A(3, 3);
    A[0][0] = 2;    A[0][1] = 1;    A[0][2] = 7;
    A[1][0] = 8;    A[1][1] = 8;    A[1][2] = 33;
    A[2][0] = -4;   A[2][1] = 10;   A[2][2] = 4;
    
    PRINT(b);
    PRINT(A);

    hdnum::Vector<hdnum::FP256> x(3, 0.0); // holds the solution
    hdnum::Vector<hdnum::FP256> s(3); // 
    hdnum::Vector<std::size_t> p(3);
    hdnum::Vector<std::size_t> q(3);


    PRINT(x);
    PRINT(s);
    PRINT(p);
    PRINT(q);

    hdnum::row_equilibrate(A, s);
    hdnum::lr_fullpivot(A, p, q);

    PRINT(A);
    PRINT(p);
    PRINT(q);

    hdnum::apply_equilibrate(s, b);
    hdnum::permute_forward(p, b);

    hdnum::solveL(A, b, b);
    hdnum::solveR(A, x, b);

    hdnum::permute_backward(q, x);

    PRINT(x);


    






    return 0;
}