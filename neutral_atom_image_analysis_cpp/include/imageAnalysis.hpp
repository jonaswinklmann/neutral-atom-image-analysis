#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace py = pybind11;

class ImageAnalysis
{
protected:
    const pybind11::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> psf;
    std::vector<std::tuple<double, double>> atomLocations;
public:
    ImageAnalysis(const pybind11::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& psf, 
        std::vector<std::tuple<double, double>> atomLocations) : psf(psf), atomLocations(atomLocations) 
    {};
    virtual std::vector<double> reconstruct(py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> image) = 0;
};