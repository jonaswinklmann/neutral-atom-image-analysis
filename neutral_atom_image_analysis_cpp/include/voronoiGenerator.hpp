#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <tuple>

namespace py = pybind11;

class VoronoiGenerator
{
public:
    VoronoiGenerator(){};
    bool generate(py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> image,
        py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> voronoi,
        std::vector<std::tuple<double, double>> atomLocations, std::vector<std::tuple<double, double>> potentialAtomLocations,
        double psfDistanceMult, int rowPadding, int colPadding);
};