#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <optional>
#include <vector>
#include <tuple>

#include <fstream>

#include "imageAnalysis.hpp"

struct Image
{
    const double *image;
    size_t offset, outerStride, innerStride;
    // Rounded PSF center coordinates.
    int X_int, Y_int;
    // Rounded PSF rectangle corners.
    int X_min, X_max, Y_min, Y_max;
    // Subpixel shifts.
    int dx, dy;
};

namespace py = pybind11;

class ImageAnalysisProjection : ImageAnalysis
{
protected:
    int psfSupersample;
    Eigen::Array2i projShape;
    std::vector<std::vector<double>> imageProjs;

    std::vector<Image> getLocalImages(
        const py::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& fullImage);
    std::vector<double> applyProjectors(std::vector<Image>& localImages);
public:
    ImageAnalysisProjection(const pybind11::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& psf, 
        std::vector<std::tuple<double, double>> atomLocations) : ImageAnalysis(psf, atomLocations)
    {};
    std::vector<double> reconstruct(py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> image) override;
    void setProjGen(py::object& prjgen);
};