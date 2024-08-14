// #include <pybind11/pybind11.h>
// #include <biest.hpp> // BIEST
// #include <sctl.hpp>  // BIEST's dependency
// #include <xtensor/xarray.hpp>
// #include <xtensor-python/pyarray.hpp>
// typedef xt::pyarray<double> Array;
// typedef double Real;
// typedef sctl::Vector<biest::Surface<Real>> Surface;

// // Function to create a 2x3 pyarray
// xt::pyarray<double> create_2x3_pyarray()
// {
//     // Define the shape as a vector
//     std::vector<std::size_t> shape = {2, 3};

//     // Create an uninitialized pyarray with the given shape
//     xt::pyarray<double> arr = xt::empty<double>(shape);

//     // Optionally, fill the array with some values
//     for (std::size_t i = 0; i < arr.shape(0); ++i)
//     {
//         for (std::size_t j = 0; j < arr.shape(1); ++j)
//         {
//             arr(i, j) = static_cast<double>(i * arr.shape(1) + j);
//         }
//     }

//     return arr;
// }

// static void test()
// {
//     constexpr int DIM = 3; // dimensions of coordinate space
//     const int digits = 10; // number of digits of accuracy requested

//     const int NFP = 1, Nt = 70, Np = 20;
//     sctl::Vector<Real> X(DIM * Nt * Np), F(Nt * Np), U;
//     for (int i = 0; i < Nt; i++)
//     { // initialize data X, F
//         for (int j = 0; j < Np; j++)
//         {
//             const Real phi = 2 * sctl::const_pi<Real>() * i / Nt;
//             const Real theta = 2 * sctl::const_pi<Real>() * j / Np;

//             const Real R = 1 + 0.25 * sctl::cos<Real>(theta);
//             const Real x = R * sctl::cos<Real>(phi);
//             const Real y = R * sctl::sin<Real>(phi);
//             const Real z = 0.25 * sctl::sin<Real>(theta);

//             X[(0 * Nt + i) * Np + j] = x;
//             X[(1 * Nt + i) * Np + j] = y;
//             X[(2 * Nt + i) * Np + j] = z;
//             F[i * Np + j] = x + y + z;
//         }
//     }

//     // const auto kernel = biest::Laplace3D<Real>::FxU(); // Laplace single-layer kernel function
//     const auto kernel = biest::Laplace3D<Real>::DxU(); // Laplace double-layer kernel function
//     constexpr int KER_DIM0 = 1;                        // input degrees-of-freedom of kernel
//     constexpr int KER_DIM1 = 1;                        // output degrees-of-freedom of kernel

//     biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator

//     Surface Svec(1);
//     Svec[0] = biop.BuildSurface(X, NFP, Nt, Np); // build surface object

//     biop.SetupSingular(Svec, kernel, digits, NFP, Nt, Np, Nt, Np); // initialize biop
//     biop.Eval(U, F, NFP, Nt, Np);                                  // evaluate potential

//     WriteVTK("F", Svec, F); // visualize F
//     WriteVTK("U", Svec, U); // visualize U
// }

// // void test_array()
// // {
// //     constexpr int DIM = 3; // dimensions of coordinate space
// //     const int digits = 10; // number of digits of accuracy requested

// //     const int NFP = 1, Nt = 70, Np = 20;
// //     sctl::Vector<Real> X(DIM * Nt * Np), F(Nt * Np), U;
// //     for (int i = 0; i < Nt; i++)
// //     { // initialize data X, F
// //         for (int j = 0; j < Np; j++)
// //         {
// //             const Real phi = 2 * sctl::const_pi<Real>() * i / Nt;
// //             const Real theta = 2 * sctl::const_pi<Real>() * j / Np;

// //             const Real R = 1 + 0.25 * sctl::cos<Real>(theta);
// //             const Real x = R * sctl::cos<Real>(phi);
// //             const Real y = R * sctl::sin<Real>(phi);
// //             const Real z = 0.25 * sctl::sin<Real>(theta);

// //             X[(0 * Nt + i) * Np + j] = x;
// //             X[(1 * Nt + i) * Np + j] = y;
// //             X[(2 * Nt + i) * Np + j] = z;
// //             F[i * Np + j] = x + y + z;
// //         }
// //     }

// //     // const auto kernel = biest::Laplace3D<Real>::FxU(); // Laplace single-layer kernel function
// //     const auto kernel = biest::Laplace3D<Real>::DxU(); // Laplace double-layer kernel function
// //     constexpr int KER_DIM0 = 1;                        // input degrees-of-freedom of kernel
// //     constexpr int KER_DIM1 = 1;                        // output degrees-of-freedom of kernel

// //     biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator

// //     Surface Svec(1);
// //     Svec[0] = biop.BuildSurface(X, NFP, Nt, Np); // build surface object

// //     biop.SetupSingular(Svec, kernel, digits, NFP, Nt, Np, Nt, Np); // initialize biop
// //     biop.Eval(U, F, NFP, Nt, Np);                                  // evaluate potential

// //     Array U_arr = xt::empty<double>({Nt, Np});
// //     for (int i = 0; i < Nt; i++)
// //     { // initialize data X, F
// //         for (int j = 0; j < Np; j++)
// //         {
// //             U_arr[i, j] = U[i * Np + j];
// //         }
// //     }
// //     // return U_arr;
// // }

// /*

// Array biest_call(Array gamma, Array f_arr, int digits, int nfp, int nt, int np)
// {
//     // #pragma omp parallel for
//     constexpr int DIM = 3; // dimensions of coordinate space
//     // Loading surface shape
//     sctl::Vector<Real> X(DIM * nt * np);
//     for (int i = 0; i < nt; i++)
//     { // initialize data X, F
//         for (int j = 0; j < np; j++)
//         {
//             X[(0 * nt + i) * np + j] = gamma[i, j, 0];
//             X[(1 * nt + i) * np + j] = gamma[i, j, 1];
//             X[(2 * nt + i) * np + j] = gamma[i, j, 2];
//         }
//     }

//     // Building Kernel
//     if (single_layer_mode)
//     {
//         const auto kernel = biest::Laplace3D<Real>::FxU(); // Laplace single-layer kernel function
//     }
//     else
//     {
//         const auto kernel = biest::Laplace3D<Real>::DxU(); // Laplace double-layer kernel function
//     }
//     constexpr int KER_DIM0 = 1;                        // input degrees-of-freedom of kernel
//     constexpr int KER_DIM1 = 1;                        // output degrees-of-freedom of kernel
//     biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator

//     // build surface object
//     // Relatively costly, and will only be run once.
//     Surface Svec(1);
//     Svec[0] = biop.BuildSurface(X, nfp, nt, np);
//     biop.SetupSingular(Svec, kernel, digits, nfp, nt, np, nt, np); // initialize biop

//     // Evaluating integrals
//     ndim_integrand = f_arr.shape()[-1];
//     sctl::Vector<Real> F(nt * np), U, U_arr(ndim_integrand);
//     for (int k = 0; k < ndim_integrand; k++)
//     {
//         for (int i = 0; i < nt; i++)
//         { // initialize F
//             for (int j = 0; j < np; j++)
//             {
//                 F[i * np + j] = f_arr[i, j, k];
//             }
//         }
//         biop.Eval(U, F, nfp, nt, np); // evaluate potential
//         U_arr[k] = U;
//     }
//     return U_arr;
// }
// */

// int add(int i, int j)
// {
//     return i + j;
// }

// PYBIND11_MODULE(biest_call, m)
// {
//     m.doc() = "Integrate singular functions with BIEST"; // optional module docstring
//     m.def("add", &add, "A function that adds two numbers");
//     m.def("test", &test, "The BIEST test case");
//     // m.def("test_array", &test_array, "The BIEST test case");
//     m.def("create_2x3_pyarray", &create_2x3_pyarray, "Create a 2x3 pyarray");
// }
#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>

xt::pyarray<double> example_function()
{
    xt::pyarray<double>::shape_type shape = {2, 3};
    xt::pyarray<double> arr(shape);
    arr.fill(42);
    return arr;
}

PYBIND11_MODULE(biest_call, m)
{
    m.def("example_function", &example_function, "A simple example function");
}
