#include <biest.hpp> // BIEST
#include <numeric>                    // Standard library import for std::accumulate
#include <functional>                 // Allow the user to choose kernel while reducing duplicate codes
#include <pybind11/pybind11.h>        // Pybind11 import to define Python bindings
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY            // numpy C api loading
#include <xtensor/xarray.hpp>
#include <xtensor-python/pyarray.hpp> // Numpy bindings
// Only used in sum_of_sines
#include <xtensor/xmath.hpp> // xtensor import for the C++ universal functions
namespace py = pybind11;

namespace biest_call{
    // #include <gperftools/profiler.h> // Profiler

    typedef xt::pyarray<double> Array;
    typedef double Real;
    typedef sctl::Vector<biest::Surface<Real>> Surface;

    constexpr int DIM = 3;      // dimensions of coordinate space
    constexpr int KER_DIM0 = 1; // input degrees-of-freedom of kernel
    constexpr int KER_DIM1 = 1; // output degrees-of-freedom of kernel

    Real sum_of_sines(Array &m)
    {
        auto sines = xt::sin(m); // sines does not actually hold values.
        return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
    }

    void test_single(Real a, Real b, xt::pyarray<double> &gamma, xt::pyarray<double> &f_arr, xt::pyarray<double> &result)
    {
        constexpr int DIM = 3; // dimensions of coordinate space
        const int digits = 10; // number of digits of accuracy requested

        const int nfp = 1, Nt = 70, Np = 20;
        sctl::Vector<Real> X(DIM * Nt * Np), F(Nt * Np);
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
                gamma(i, j, 0) = x;
                gamma(i, j, 1) = y;
                gamma(i, j, 2) = z;
                f_arr(i, j) = F[i * Np + j];
            }
        }
        py::print("Initialization successful.");

        constexpr int KER_DIM0 = 1;                                    // input degrees-of-freedom of kernel
        constexpr int KER_DIM1 = 1;                                    // output degrees-of-freedom of kernel
        biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
        Surface Svec(1);
        Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object
        py::print("Surface built successfully.");
        const auto kernel = biest::Laplace3D<Real>::FxU(); // Laplace single-layer kernel function
        py::print("Kernel built successfully.");
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
        py::print("Singular integrals built successfully.");
        sctl::Vector<Real> U;
        biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
        py::print("Eval successful.");
        for (int i = 0; i < Nt; i++)
        {
            for (int j = 0; j < Np; j++)
            {
                result(i, j) = U[i * Np + j];
            }
        }
    }

    void test_double(Real a, Real b, xt::pyarray<double> &gamma, xt::pyarray<double> &f_arr, xt::pyarray<double> &result)
    {
        constexpr int DIM = 3; // dimensions of coordinate space
        const int digits = 10; // number of digits of accuracy requested

        const int nfp = 1, Nt = 70, Np = 20;
        sctl::Vector<Real> X(DIM * Nt * Np), F(Nt * Np);
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
                gamma(i, j, 0) = x;
                gamma(i, j, 1) = y;
                gamma(i, j, 2) = z;
                f_arr(i, j) = F[i * Np + j];
            }
        }
        py::print("Initialization successful.");

        constexpr int KER_DIM0 = 1;                                    // input degrees-of-freedom of kernel
        constexpr int KER_DIM1 = 1;                                    // output degrees-of-freedom of kernel
        biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
        Surface Svec(1);
        Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object
        py::print("Surface built successfully.");
        const auto kernel = biest::Laplace3D<Real>::DxU(); // Laplace double-layer kernel function
        py::print("Kernel built successfully.");
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
        py::print("Singular integrals built successfully.");
        sctl::Vector<Real> U;
        biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
        py::print("Eval successful.");
        for (int i = 0; i < Nt; i++)
        {
            for (int j = 0; j < Np; j++)
            {
                result(i, j) = U[i * Np + j];
            }
        }
    }

    /*
    xt::pyarray<double> &gamma:
    r(theta, zeta) of the surface, corresponds to Surface.gamma() in simsopt.
    Has shape [n_phi, n_theta, 3]

    xt::pyarray<double> &func_in, xt::pyarray<double> &func_in_double:
    f(theta, zeta), a scalar function to multiply with a single/double layer
    Laplacian kernel and integrate over the surface. Has shape [n_phi, n_theta, dim]
    dim must be at least [n_phi, n_theta, n].

    bool single:
    If true, use the single layer kernel 1/|y-x|. Otherwise, uses the double-layer kernel.

    int digits:
    Number of digits.

    int nfp:
    Number of field periods.

    bool undo_flip: 
    BIEST automatically flips handedness when the normal vector is detected to 
    point inward. This can be disadvantageous when integrating a vector proportional
    to the normal vector using the double layer kernel, because it can break the 
    handedness-independence of some results. Enable to add a sign flip when the 
    normal is flipped. Only for double-layered kernel.
    */
    static void integrate_multi(
        xt::pyarray<double> &gamma,
        xt::pyarray<double> &func_in,
        xt::pyarray<double> &result,
        bool single,
        int digits,
        int nfp,
        bool undo_flip)
    {

        // Checking shapes
        if (func_in.dimension() != 3)
        {
            throw std::invalid_argument("func_in has invalid shape.");
        }
        if (func_in.shape(0) != result.shape(0) || func_in.shape(1) != result.shape(1) || func_in.shape(2) != result.shape(2))
        {
            throw std::invalid_argument("func_in and result has different shapes.");
        }
        if (func_in.shape(0) != gamma.shape(0) || func_in.shape(1) != gamma.shape(1) || gamma.shape(2) != 3)
        {
            throw std::invalid_argument("gamma has invalid shape.");
        }
        const int Nvec = func_in.shape(2);
        const int Nt = func_in.shape(0);
        const int Np = func_in.shape(1);

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
        biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
        Surface Svec(1);
        Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object
        if (single)
        {
            const auto kernel = biest::Laplace3D<Real>::FxU();             // Laplace single-layer kernel function
            biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
#pragma omp parallel for
            for (int k = 0; k < Nvec; k++)
            {
                sctl::Vector<Real> F(Nt * Np), U;
                for (int i = 0; i < Nt; i++)
                {
                    for (int j = 0; j < Np; j++)
                    {
                        F[i * Np + j] = func_in(i, j, k);
                    }
                }
                biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
                for (int i = 0; i < Nt; i++)
                {
                    for (int j = 0; j < Np; j++)
                    {
                        result(i, j, k) = U[i * Np + j];
                    }
                }
            }
        }
        else
        {
            // Detecting whether BIEST has flipped the handedness of the surface.
            int sign = 1;
            const auto X_temp = Svec[0].Coord();
            sctl::Vector<Real> dX, n0, n1;
            biest::SurfaceOp<Real> surf_op(sctl::Comm::Self(), Nt, Np);
            surf_op.Grad2D(dX, X_temp);
            surf_op.SurfNormalAreaElem(&n0, nullptr, dX, &X_temp); // outward-normal
            surf_op.SurfNormalAreaElem(&n1, nullptr, dX, nullptr); // cross-product normal
            const Real n0_dot_n1 = n0[Nt * Np * 0] * n1[Nt * Np * 0] + n0[Nt * Np * 1] * n1[Nt * Np * 1] + n0[Nt * Np * 2] * n1[Nt * Np * 2];
            if (undo_flip && n0_dot_n1 < 0) // Sign flip detected
            {
                py::print("Sign flip detected in BIEST during double layer integral. Undoing flip.");
                sign = -1;
            }
            // Constructing kernel and setting up the integral
            const auto kernel = biest::Laplace3D<Real>::DxU();             // Laplace double-layer kernel function
            biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
#pragma omp parallel for
            for (int k = 0; k < Nvec; k++)
            {
                sctl::Vector<Real> F(Nt * Np), U;
                for (int i = 0; i < Nt; i++)
                {
                    for (int j = 0; j < Np; j++)
                    {
                        F[i * Np + j] = func_in(i, j, k);
                    }
                }
                biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
                for (int i = 0; i < Nt; i++)
                {
                    for (int j = 0; j < Np; j++)
                    {
                        result(i, j, k) = sign * U[i * Np + j];
                    }
                }
            }
        }
    }

    static void plot_in_vtk(
        xt::pyarray<double> &gamma,
        int digits,
        int nfp)
    {
        const int Nt = gamma.shape(0);
        const int Np = gamma.shape(1);
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
        biest::FieldPeriodBIOp<Real, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
        Surface Svec(1);
        Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object
        WriteVTK("S", Svec); // Write to file for visualization
    }

    PYBIND11_MODULE(biest_call, m)
    {
        xt::import_numpy();
        m.doc() = "Test module for xtensor python bindings";
        m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
        m.def("test_single", test_single, "Testing 100 BIEST calls.");
        m.def("test_double", test_double, "Testing 100 BIEST calls.");
        m.def("integrate_multi", integrate_multi, "Integrating multiple scalar functions using BIEST");
        m.def("plot_in_vtk", plot_in_vtk, "Integrating multiple scalar functions using BIEST");
    }
}