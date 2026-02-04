#include "voronoiGenerator.hpp"

#include <cfloat>

bool VoronoiGenerator::generate(py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> image,
    py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> voronoi, 
    std::vector<std::tuple<double, double>> atomLocations, std::vector<std::tuple<double, double>> potentialAtomLocations, 
    double psfDistanceMult, int rowPadding, int colPadding)
{
    if(image.rows() != voronoi.rows() || image.cols() != voronoi.cols())
    {
        return false;
    }

    std::vector<std::tuple<double, double>> allPotentiallyOccupiedAtomLocations(atomLocations);
    if(!potentialAtomLocations.empty())
    {
        allPotentiallyOccupiedAtomLocations.insert(allPotentiallyOccupiedAtomLocations.end(),
            potentialAtomLocations.begin(), potentialAtomLocations.end());
    }
    
    #pragma omp parallel for schedule(dynamic, 8)
    for(Eigen::Index row = 0; row < image.rows(); row++)
    {
        for(Eigen::Index col = 0; col < image.cols(); col++)
        {
            size_t minDistIndex = 0;
            double minDistSq = DBL_MAX;
            double secondMinDistSq = DBL_MAX;
            bool onlyPotentialAtomSiteClosest = false;
            for(size_t atomLocationIndex = 0; atomLocationIndex < allPotentiallyOccupiedAtomLocations.size(); atomLocationIndex++)
            {
                auto [atomRow, atomCol] = allPotentiallyOccupiedAtomLocations[atomLocationIndex];
                atomRow += rowPadding;
                atomCol += colPadding;
                double distSq = (atomRow - row) * (atomRow - row) + (atomCol - col) * (atomCol - col);
                if(distSq < minDistSq)
                {
                    if(atomLocationIndex >= atomLocations.size())
                    {
                        onlyPotentialAtomSiteClosest = true;
                        break;
                    }
                    secondMinDistSq = minDistSq;
                    minDistSq = distSq;
                    minDistIndex = atomLocationIndex;
                }
                else if(distSq < secondMinDistSq)
                {
                    secondMinDistSq = distSq;
                }
            }
            if(onlyPotentialAtomSiteClosest || secondMinDistSq < minDistSq * psfDistanceMult)
            {
                voronoi(row, col) = -1;
            }
            else
            {
                voronoi(row, col) = minDistIndex;
            }
        }
    }

    return true;
}