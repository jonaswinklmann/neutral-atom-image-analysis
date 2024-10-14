#include "imageAnalysisProjection.hpp"

#include <pybind11/numpy.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <omp.h>
#include <iostream>

int ImageAnalysisProjection::setProjGen(py::object& prjgen)
{
    bool projCacheBuilt = prjgen.attr("proj_cache_built").cast<bool>();
    if(!projCacheBuilt)
    {
        prjgen.attr("setup_cache")();
    }
    this->psfSupersample = prjgen.attr("psf_supersample").cast<int>();
    this->projShape = prjgen.attr("proj_shape").cast<Eigen::Array2i>();

    py::array_t<double, py::array::c_style | py::array::forcecast> projs = prjgen.attr("proj_cache").cast<py::array_t<double>>();
    const pybind11::ssize_t *shape = projs.shape();
    if(projs.ndim() < 2)
    {
        std::cout << "Projection cache does not have sufficiently many dimensions" << std::endl;
        return -1;
    }
    projs = projs.reshape(std::vector<int>({(int)(shape[0]), (int)(shape[1]), -1}));
    const pybind11::ssize_t *newShape = projs.shape();
    if(shape[0] < this->psfSupersample || shape[1] < this->psfSupersample)
    {
        std::cout << "Projection cache dimensions 0 or 1 are smaller than psf_supersample" << std::endl;
        return -2;
    }

    for(int yidx = 0; yidx < this->psfSupersample; yidx++)
    {
        for(int xidx = 0; xidx < this->psfSupersample; xidx++)
        {
            std::vector<double> imageProj;

            py::array proj = projs[py::make_tuple(xidx, yidx, py::ellipsis())];
            py::buffer_info info = proj.request();
            double *ptr = static_cast<double*>(info.ptr);
            imageProj.insert(imageProj.end(), &ptr[0], &ptr[newShape[2]]);

            double projSum = std::accumulate(imageProj.begin(), imageProj.end(), 0.0);
            this->imageProjs.push_back(std::pair<const std::vector<double>,double>(std::move(imageProj), projSum));
        }
    }
    return 0;
}

std::vector<double> ImageAnalysisProjection::reconstruct(
    py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> image)
{
    auto localImages = this->getLocalImages(image);
    auto localEmissions = this->applyProjectors(localImages);
    return localEmissions;
}

std::vector<Image> ImageAnalysisProjection::getLocalImages(
    const py::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& fullImage)
{
    // Extracts image subregions and subpixel shifts.

    std::vector<Image> localImages;
    localImages.resize(this->atomLocations.size());

#pragma omp parallel for schedule(dynamic,8) shared(localImages)
    for(size_t i = 0; i < this->atomLocations.size(); i++)
    {
        const auto& coord = this->atomLocations[i];
        int y_int = (int)std::round(std::get<0>(coord));
        int x_int = (int)std::round(std::get<1>(coord));
        int y_min = y_int - this->projShape[0] / 2;
        int x_min = x_int - this->projShape[1] / 2;
        int y_max = y_min + this->projShape[0] - 1;
        int x_max = x_min + this->projShape[1] - 1;
        Image imageN
        {
            .image = fullImage.data(),
            .imageRows = fullImage.rows(),
            .imageCols = fullImage.cols(),
            .outerStride = (size_t)(fullImage.cols()),
            .innerStride = 1,
            .X_int = x_int,
            .Y_int = y_int,
            .X_min = x_min,
            .X_max = x_max,
            .Y_min = y_min,
            .Y_max = y_max,
            .dx = (int)(std::round((std::get<1>(coord) - x_int) * this->psfSupersample)),
            .dy = (int)(std::round((std::get<0>(coord) - y_int) * this->psfSupersample))
        };
        localImages[i] = imageN;
    }
    return localImages;
}

std::vector<double> ImageAnalysisProjection::applyProjectors(std::vector<Image>& localImages)
{
    std::vector<double> emissions;
    emissions.resize(localImages.size());

#pragma omp parallel for schedule(dynamic,8) shared(emissions)
    for(size_t i = 0; i < localImages.size(); i++)
    {
        const Image& localImage = localImages[i];
        int xidx = (localImage.dx + this->psfSupersample) % this->psfSupersample;
        int yidx = (localImage.dy + this->psfSupersample) % this->psfSupersample;
        if ((unsigned int)(yidx * this->psfSupersample + xidx) >= imageProjs.size())
        {
            std::cout << "Not enough image projections in cache list. Skipping atom site" << std::endl;
            continue;
        }
        const auto& imageProj = imageProjs[yidx * this->psfSupersample + xidx].first;
        double imageProjSum = imageProjs[yidx * this->psfSupersample + xidx].second;

        double projSumUsed = 0;
        double sum = 0;
        int cols = localImage.X_max - localImage.X_min + 1;
        if ((unsigned int)(cols * (localImage.Y_max - localImage.Y_min + 1)) > imageProj.size())
        {
            std::cout << "Not enough values in projection data. Skipping atom site" << std::endl;
            continue;
        }

        int row = localImage.Y_min;
        if(row < 0)
        {
            row = 0;
        }
        for(; row <= localImage.Y_max && row < localImage.imageRows; row++)
        {
            size_t p = (row - localImage.Y_min) * cols;
            int col = localImage.X_min;
            if(col < 0)
            {
                p -= col;
                col = 0;
            }
            for(; col <= localImage.X_max && col < localImage.imageCols; col++)
            {
                sum += localImage.image[row * localImage.outerStride + col] * imageProj[p];
                projSumUsed += imageProj[p];
                p++;
            }
        }
        if(projSumUsed > 0)
        {
            emissions[i] = sum * (imageProjSum / projSumUsed);
        }
        else
        {
            emissions[i] = 0;
        }
    }
    return emissions;
}