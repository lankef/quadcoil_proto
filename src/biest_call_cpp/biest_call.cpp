#include <stdexcept>
#include <biest.hpp> // BIEST
#include <sctl.hpp>  // BIEST's dependency

#include <numeric>                    // Standard library import for std::accumulate
#include <pybind11/pybind11.h>        // Pybind11 import to define Python bindings
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY            // numpy C api loading
#include <xtensor/xarray.hpp>
#include <xtensor-python/pyarray.hpp> // Numpy bindings

typedef xt::pyarray<double> Array;
typedef double Real;
typedef sctl::Vector<biest::Surface<Real>> Surface;

// Only used in sum_of_sines
#include <xtensor/xmath.hpp> // xtensor import for the C++ universal functions

namespace py = pybind11;

static double sum_of_sines(xt::pyarray<double> &m)
{
    auto sines = xt::sin(m); // sines does not actually hold values.
    return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
}

static void edit_2x3_pyarray(xt::pyarray<double> &arr)
{
    for (std::size_t i = 0; i < arr.shape(0); ++i)
    {
        for (std::size_t j = 0; j < arr.shape(1); ++j)
        {
            arr(i, j) = static_cast<double>(i * arr.shape(1) + j);
        }
    }
}

static double test(bool single, double a, double b)
{
    constexpr int DIM = 3; // dimensions of coordinate space
    const int digits = 10; // number of digits of accuracy requested

    const int nfp = 1, Nt = 70, Np = 20;
    sctl::Vector<Real> X(DIM * Nt * Np), F(Nt * Np), U;
    for (int i = 0; i < Nt; i++)
    { // initialize data X, F
        for (int j = 0; j < Np; j++)
        {
            const Real phi = 2 * sctl::const_pi<Real>() * i / Nt;
            const Real theta = 2 * sctl::const_pi<Real>() * j / Np;

            const Real R = 1 + 0.25 * sctl::cos<Real>(theta);
            const Real x = R * sctl::cos<Real>(phi);
            const Real y = R * a * sctl::sin<Real>(phi);
            const Real z = 0.25 * sctl::sin<Real>(theta);

            X[(0 * Nt + i) * Np + j] = x;
            X[(1 * Nt + i) * Np + j] = y;
            X[(2 * Nt + i) * Np + j] = z;
            F[i * Np + j] = x + y + b * z;
        }
    }

    constexpr int KER_DIM0 = 1; // input degrees-of-freedom of kernel
    constexpr int KER_DIM1 = 1; // output degrees-of-freedom of kernel
    biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
    Surface Svec(1);
    Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object

    if (single){
        const auto kernel = biest::Laplace3D<Real>::FxU(); // Laplace single-layer kernel function
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
    } else {
        const auto kernel = biest::Laplace3D<Real>::DxU(); // Laplace double-layer kernel function
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
    }
    
    biop.Eval(U, F, nfp, Nt, Np);                                  // evaluate potential

    WriteVTK("F", Svec, F); // visualize F
    WriteVTK("U", Svec, U); // visualize U
    double out = U[0];
    return out;
}
/*
xt::pyarray<double> &gamma:
r(theta, zeta) of the surface, corresponds to Surface.gamma() in simsopt.
Has shape [n_phi, n_theta, 3]

xt::pyarray<double> &func_in:
f(theta, zeta), a scalar function to multiply with a Laplacian kernel and
integrate over the surface. Has shape [n_phi, n_theta]

int digits:
Number of digits.

int nfp:
Number of field periods.

bool single_layer:
Whether to use the single layer kernel (true) or double layer kernel (false).
*/
static double integrate_scalar(xt::pyarray<double> &gamma, xt::pyarray<double> &func_in, int digits, int nfp, bool single_layer)
{
    constexpr int DIM = 3; // dimensions of coordinate space
    // const int digits = 10; // number of digits of accuracy requested
    // const int nfp = 1, Nt = 70, Np = 20;
    int Nt = func_in.shape(0);
    int Np = func_in.shape(1);
    sctl::Vector<Real> X(DIM * Nt * Np), F(Nt * Np), U;
    for (int i = 0; i < Nt; i++)
    { // initialize data X, F
        for (int j = 0; j < Np; j++)
        {
            X[(0 * Nt + i) * Np + j] = gamma(i, j, 0); // x
            X[(1 * Nt + i) * Np + j] = gamma(i, j, 1); // y
            X[(2 * Nt + i) * Np + j] = gamma(i, j, 2); // z
            F[i * Np + j] = func_in(i, j);
        }
    }

    // Constructing surface and integral
    constexpr int KER_DIM0 = 1;                                    // input degrees-of-freedom of kernel
    constexpr int KER_DIM1 = 1;                                    // output degrees-of-freedom of kernel
    biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
    Surface Svec(1);
    Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object
    if (single_layer)
    {
        // Laplace single-layer kernel function
        const auto kernel = biest::Laplace3D<Real>::FxU();
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
    }
    else
    {
        // Laplace double-layer kernel function
        const auto kernel = biest::Laplace3D<Real>::DxU();
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
    }
    // Evaluating the integral
    biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
    double out = U[0];
    return out;
}

/*
xt::pyarray<double> &gamma:
r(theta, zeta) of the surface, corresponds to Surface.gamma() in simsopt.
Has shape [n_phi, n_theta, 3]

xt::pyarray<double> &func_in_single, xt::pyarray<double> &func_in_double:
f(theta, zeta), a scalar function to multiply with a single/double layer
Laplacian kernel and integrate over the surface. Has shape [n_phi, n_theta, dim]
dim must be at least 1.

int digits:
Number of digits.

int nfp:
Number of field periods.

bool single_layer:
Whether to use the single layer kernel (true) or double layer kernel (false).
*/
static void integrate_multi(
    xt::pyarray<double> &gamma,
    xt::pyarray<double> &func_in_single,
    xt::pyarray<double> &func_in_double,
    xt::pyarray<double> &result,
    int digits,
    int nfp)
{
    constexpr int DIM = 3; // dimensions of coordinate space

    // Because constructing the integral is costly, 
    // we add the option of skipping either the single,
    // or the double-layer integrals by giving a 0-d array.
    // Either of func_in_single can be zero dimensional.
    int Nt = 0;
    int Np = 0;
    int Nsingle = 0;
    int Ndouble = 0;
    // Checking shapes
    if (func_in_single.dimension() != 0)
    {
        if (func_in_single.dimension() != 3)
        {
            throw std::invalid_argument("func_in_single has invalid shape!");
        }
        Nsingle = func_in_single.shape(2);
        Nt = func_in_single.shape(0);
        Np = func_in_single.shape(1);
    }
    else
    {
        Nsingle = 0;
    }
    if (func_in_double.dimension() != 0)
    {
        if (func_in_double.dimension() != 3)
        {
            throw std::invalid_argument("func_in_double has invalid shape!");
        }
        Ndouble = func_in_double.shape(2);
        Nt = func_in_double.shape(0);
        Np = func_in_double.shape(1);
    }
    else
    {
        Ndouble = 0;
    }
    int Nresult = Nsingle + Ndouble;
    if (result.dimension() != 1){
        throw std::invalid_argument("result must be an 1d array");
    }
    if (result.shape(0) != Nresult)
    {
        throw std::invalid_argument("The length of result must be equal to the sum of the length of the axis=2 of both func_in.");
    }

    // Detecting improperly shaped arrays
    if (Nsingle != 0 && Ndouble != 0){
        if ((func_in_single.shape(0) != func_in_double.shape(0) || func_in_single.shape(1) != func_in_double.shape(1)))
        {
            throw std::invalid_argument("The first 2 dimensions of func_in_single and func_in_double has different length");
        }
    }

    // Loading gamma
    sctl::Vector<Real> X(DIM * Nt * Np);
    for (int i = 0; i < Nt; i++)
    { // initialize data X
        for (int j = 0; j < Np; j++)
        {
            X[(0 * Nt + i) * Np + j] = gamma(i, j, 0); // x
            X[(1 * Nt + i) * Np + j] = gamma(i, j, 1); // y
            X[(2 * Nt + i) * Np + j] = gamma(i, j, 2); // z
        }
    }

    // Constructing the surface.
    constexpr int KER_DIM0 = 1;                                    // input degrees-of-freedom of kernel
    constexpr int KER_DIM1 = 1;                                    // output degrees-of-freedom of kernel
    Surface Svec(1);
    if (Nsingle != 0)
    {
        // Laplace single-layer kernel function
        const auto kernel = biest::Laplace3D<Real>::FxU();
        biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
        Svec[0] = biop.BuildSurface(X, nfp, Nt, Np);                   // build surface object
        // Expensive
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np);
        for (int k = 0; k < Nsingle; k++)
        {
            sctl::Vector<Real> F(Nt * Np), U;
            // initialize data F
            for (int i = 0; i < Nt; i++)
            {
                for (int j = 0; j < Np; j++)
                {
                    F[i * Np + j] = func_in_single(i, j, k);
                }
            }
            biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
            py::print("Integral #", k, "=", U[0]); // printing the integration results
            result(k) = U[0];
        }
    }

    if (Ndouble != 0)
    {
        // Laplace double-layer kernel function
        const auto kernel = biest::Laplace3D<Real>::DxU();
        biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
        Svec[0] = biop.BuildSurface(X, nfp, Nt, Np);                   // build surface object
        // Expensive
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np);
        for (int k = 0; k < Ndouble; k++)
        {
            sctl::Vector<Real> F(Nt * Np), U;
            // initialize data F
            for (int i = 0; i < Nt; i++)
            {
                for (int j = 0; j < Np; j++)
                {
                    F[i * Np + j] = func_in_double(i, j, k);
                }
            }
            biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
            py::print("Integral #", k, "=", U[0]); // Printing the integration results
            result(Nsingle + k) = U[0];
        }
    }
}

static bool is_zero_d(xt::pyarray<double> &arr){
    return arr.dimension() == 0;
}

PYBIND11_MODULE(biest_call, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";
    m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
    m.def("is_zero_d", is_zero_d, "Can scalars be passed as 0d arrays");
    m.def("edit_2x3_pyarray", edit_2x3_pyarray, "Test that creates arrays");
    m.def("test", test, "Testing BIEST");
    m.def("integrate_scalar", integrate_scalar, "Integrating a scalar function using BIEST");
    m.def("integrate_multi", integrate_multi, "Integrating a scalar function using BIEST");
}
