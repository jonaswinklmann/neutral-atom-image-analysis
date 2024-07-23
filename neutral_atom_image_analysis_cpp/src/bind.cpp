#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "imageAnalysisProjection.hpp"

namespace py = pybind11;

PYBIND11_MODULE(neutral_atom_image_analysis_cpp, m) {
    m.doc() = "pybind11 neutral_atom_image_analysis_cpp module";

    py::class_<ImageAnalysisProjection>(m, "ImageAnalysisProjection")
        .def(py::init<const py::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&, 
            std::vector<std::tuple<double, double>>>())
        .def("reconstruct", &ImageAnalysisProjection::reconstruct, py::arg("image"))
        .def("setProjGen", &ImageAnalysisProjection::setProjGen, py::arg("prjgen"));
}