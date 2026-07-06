#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "imageAnalysisProjection.hpp"
#include "voronoiGenerator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(neutral_atom_image_analysis_cpp, m) {
    m.doc() = "pybind11 neutral_atom_image_analysis_cpp module";

    py::class_<ImageAnalysisProjection>(m, "ImageAnalysisProjection", R"pbdoc(
            A ImageAnalysis child class for projection-based image analysis.
        )pbdoc")
        .def(py::init<const py::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&, 
            std::vector<std::tuple<double, double>>>())
        .def("reconstruct", &ImageAnalysisProjection::reconstruct, py::arg("image"))
        .def("setProjectors", &ImageAnalysisProjection::setProjectors, py::arg("prjgen"))
        .def("setProjectorsFromArray", &ImageAnalysisProjection::setProjectorsFromArray, py::arg("projectors"));
    
    py::class_<VoronoiGenerator>(m, "VoronoiGenerator")
        .def(py::init<>())
        .def("generate", &VoronoiGenerator::generate, py::arg("image"), py::arg("voronoi"), 
            py::arg("atomLocations"), py::arg("potentialAtomLocations"), py::arg("psfDistanceMult"), 
            py::arg("rowPadding"), py::arg("colPadding"));
}