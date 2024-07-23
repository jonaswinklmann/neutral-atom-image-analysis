"""
Neutral atom image analysis wrapper
"""

import sys
import os
if sys.platform == "win32":
    os.add_dll_directory(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__) + "/")
import neutral_atom_image_analysis_cpp
import state_reconstruction
from libics.tools.trafo.linear import AffineTrafo2d
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter
import numpy as np
import abc

class ImageAnalysis(abc.ABC):
    def __init__(self, psf, atom_locations, spacing, angle):
        self.psf = psf
        self.atom_locations = atom_locations
        self.spacing = spacing
        self.angle = angle
    
    @abc.abstractmethod
    def calibrate(self):
        return
    
    @abc.abstractmethod
    def reconstruct(self, image):
        return
    

class ImageAnalysisProjection(ImageAnalysis):
    def __init__(self, psf, atom_locations, spacing, angle, psf_supersample = 1):
        ImageAnalysis.__init__(self, psf, atom_locations, spacing, angle)
        self.solver = neutral_atom_image_analysis_cpp.ImageAnalysisProjection(psf, atom_locations)
        self.psf_supersample = psf_supersample

    def calibrate(self):
        site_ref = [0,0]
        image_ref = [int(self.atom_locations[0][0]), int(self.atom_locations[0][1])]

        trafo_site_to_image = AffineTrafo2d()
        # Set site unit vectors within image coordinate system

        if np.isscalar(self.spacing):
            mag = [self.spacing,self.spacing]
        else:
            mag = self.spacing
        if np.isscalar(self.angle):
            angle = [self.angle,self.angle]
        elif self.angle is None:
            angle = [0,0]
        else:
            angle = self.angle

        # Temp solution, find better way of aquiring magnification
        trafo_site_to_image.set_origin_axes(
            magnification=mag,
            angle=angle
        )
        trafo_site_to_image.set_offset_by_point_pair(
            site_ref, image_ref
        )

        ipsf_gen = state_reconstruction.IntegratedPsfGenerator(
            psf=self.psf, psf_supersample=self.psf_supersample
        )

        proj_gen = state_reconstruction.ProjectorGenerator(
            trafo_site_to_image=trafo_site_to_image,
            integrated_psf_generator=ipsf_gen,
            proj_shape=(61, 61)
        )

        self.solver.setProjGen(proj_gen)

    def reconstruct(self, image):
        # Preprocess image
        image = np.array(image, dtype=float)
        if np.isfortran(image):
            image = np.ascontiguousarray(image)
        
        parameters = self.solver.reconstruct(image)
        return parameters