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
import matplotlib.pyplot as plt
from libics.tools import plot
from libics.tools.trafo.linear import AffineTrafo2d
from pandas import DataFrame
from skimage.segmentation import watershed
from skimage.transform import radon
from scipy.interpolate import interp1d
from scipy.ndimage import zoom, shift
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.signal import find_peaks
import numpy as np
import abc

def three_gaussian_peaks(x, loc, scale, o_scale1, o_scale2, o_scale3, offset):
    return offset + norm.pdf(x, loc = loc, scale = scale) * o_scale1 + \
        norm.pdf(x, loc = loc * 2, scale = scale) * o_scale2 + \
        norm.pdf(x, loc = loc * 3, scale = scale) * o_scale3

def two_gaussians(x, loc1, scale1, w1, loc2, scale2, w2):
    return norm.pdf(x, loc = loc1, scale = scale1) * w1 + \
        norm.pdf(x, loc = loc2, scale = scale2) * w2

class ImageAnalysis(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def calibrate(self):
        return
    
    @abc.abstractmethod
    def reconstruct(self, image):
        return
    

class ImageAnalysisProjection(ImageAnalysis):
    def __init__(self, psf_supersample = 1):
        ImageAnalysis.__init__(self)
        self.psf_supersample = psf_supersample

    def _find_atom_locations(self, average_image, angle_guesses):
        target_axes = [90,0]
        origins_first_peak_axes = []
        dirs_first_peak_axes = []

        # First x, then y, angle with respect to target axis
        self.angle = [0,0]
        self.spacing = [0,0]
        self.sites_shape = [0,0]

        # Handle both dimensions separately starting from a rough guess of the angle
        for dim in range(2):
            # Find the best suited angle within +- 15 degrees
            tested_angles = np.linspace(angle_guesses[dim] - 15, angle_guesses[dim] + 15, 101)
            h = radon(average_image, theta=tested_angles)
            highest_var_index = np.argmax(np.var(h, axis=0))

            # Find more precise angle within +- 1 degree of approx angle
            tested_angles = np.linspace(tested_angles[highest_var_index] - 1, tested_angles[highest_var_index] + 1, 101)
            h = radon(average_image, theta=tested_angles)
            highest_var_index = np.argmax(np.var(h, axis=0))

            self.angle[dim] = target_axes[dim] - tested_angles[highest_var_index]

            # Use existing radon transform to acquire image projection at given angle
            projection = h[...,highest_var_index].flatten()
            projection -= np.median(projection)

            # Compute autocorrelation to find periodicity
            result = np.correlate(projection, projection, mode='full')
            result = result[result.size//2:]
            index = 0
            for i, v in enumerate(np.diff(result)):
                if v > 0:
                    index = i
                    break

            # Find first peak (other than x=0) and fit three gaussians to the first three peaks (to improve precision)
            index += np.argmax(result[index:])
            start_index = index // 2
            end_index = int(3.5 * index) + 1
            if end_index > len(result):
                end_index = len(result)
            peak_width_guess = 5
            peak_factor_guess = (result[index] - result[index - 10]) * peak_width_guess
            x_range = range(start_index,end_index)

            popt, _ = curve_fit(three_gaussian_peaks, x_range, result[start_index:end_index], 
                p0=[index, peak_width_guess, peak_factor_guess, peak_factor_guess, peak_factor_guess, result[start_index]])
            self.spacing[dim] = popt[0]

            # Acquire rolling sum over five adjacent elements of projection to smooth over noise
            padded_n_f = np.pad(projection, (2,2), 'constant')
            neighboring_five = padded_n_f[0:-4] + padded_n_f[1:-3] + padded_n_f[2:-2] + padded_n_f[3:-1] + padded_n_f[4:]

            # Use smoothed projection to find offset for which the values at the periodic locations are maximized on average
            max_average_value = 0
            interpolation = interp1d(np.arange(len(neighboring_five)), neighboring_five)
            for offset in np.linspace(0, self.spacing[dim], 100, endpoint=False):
                index_count = int((len(neighboring_five) - 1 - offset) / self.spacing[dim]) + 1
                indices = np.array(range(index_count)) * self.spacing[dim] + offset
                values = interpolation(indices)
                average_value = np.average(values)
                if average_value > max_average_value:
                    max_average_value = average_value
                    maximizing_values = values
                    maximizing_indices = indices

            # At the given periodic locations, take all that are at least 1/e times the maximum 
            # or that lie in the middle of locations where that is the case
            threshold = maximizing_values.max() / np.e
            for i, value in enumerate(maximizing_values):
                if value >= threshold:
                    start_index = i
                    break
            for i in range(len(maximizing_values)):
                if maximizing_values[-i - 1] >= threshold:
                    end_index = len(maximizing_values) - i
                    break
            self.sites_shape[dim] = end_index - start_index

            indices = []
            values = []
            for i in range(start_index, end_index):
                indices.append(maximizing_indices[i])
                values.append(maximizing_values[i])

            # To get a more precise subpixel location, find the offset for which the difference 
            # between projection and the given number of gaussian peaks is minimal
            def gaussian(x, loc_offset, scale):
                result = 0
                for index, value in zip(indices, values):
                    result += norm.pdf(x, loc = index + loc_offset, scale = scale) * value * np.sqrt(2 * np.pi)
                return result

            popt, _ = curve_fit(gaussian, range(len(projection)), projection, p0 = [0, 3])
            for i in range(len(indices)):
                indices[i] += popt[0]

            # Save origin and direction of axis along first row/column of atom sites
            # Saved direction is along projection axis, meaning it would be projected onto a point
            origin_loc = np.array((average_image.shape[0] // 2, average_image.shape[1] // 2)).astype(np.float64)
            proj_vector = np.array((-np.sin(np.deg2rad(tested_angles[highest_var_index])), np.cos(np.deg2rad(tested_angles[highest_var_index]))))
            origin_loc += (indices[0] - float(average_image.shape[0] // 2)) * proj_vector
            origins_first_peak_axes.append(origin_loc)
            direction = np.array([np.cos(np.deg2rad(tested_angles[highest_var_index])), np.sin(np.deg2rad(tested_angles[highest_var_index]))])
            dirs_first_peak_axes.append(direction)

        # Calculate reference atom site from each dimension's origin and direction
        directions = np.array([[dirs_first_peak_axes[0][0],-dirs_first_peak_axes[1][0]],[dirs_first_peak_axes[0][1],-dirs_first_peak_axes[1][1]]])
        offsets = np.array([origins_first_peak_axes[1][0] - origins_first_peak_axes[0][0], origins_first_peak_axes[1][1] - origins_first_peak_axes[0][1]])
        solution = np.linalg.solve(directions, offsets)

        self._image_ref = dirs_first_peak_axes[0] * solution[0] + origins_first_peak_axes[0]

        proj_vectors = [origin - self._image_ref for origin in origins_first_peak_axes]
        proj_vectors.reverse()
        proj_vectors = [p / np.linalg.norm(p) for p in proj_vectors] 

        # Calculate list of atom sites
        self.atom_locations = []
        for c in range(self.sites_shape[0]):
            for r in range(self.sites_shape[1]):
                location = self._image_ref + c * self.spacing[0] * proj_vectors[0] + r * self.spacing[1] * proj_vectors[1]
                if min(location) >= 0 and location[0] <= average_image.shape[0] - 1 and location[1] <= average_image.shape[1] - 1:
                    self.atom_locations.append(location)
    
    def _find_psf(self, images, average_closed_shutter_image):
        if self.atom_locations is None:
            raise AttributeError("Atom locations not yet set", name="atom_locations", obj=self)
        
        # Integrate over roi around atom sites to determine sites to use for PSF generation
        atom_location_masks = []
        Y, X = np.ogrid[:average_closed_shutter_image.shape[0], :average_closed_shutter_image.shape[1]]
        roi_radius = 5

        for atom_location in self.atom_locations:
            dist_from_center = np.sqrt((Y - atom_location[0])**2 + (X - atom_location[1])**2)
            mask = dist_from_center <= roi_radius
            atom_location_masks.append((atom_location, mask.copy()))

        averages = []
        for image in images:
            if isinstance(image, DataFrame):
                image_np = image.to_numpy(np.float64)
            else:
                image_np = np.array(image).astype(np.float64)
            image_np -= average_closed_shutter_image
            for atom_location, mask in atom_location_masks:
                averages.append(np.average(image_np[mask]))

        value_count = len(images) * len(self.atom_locations)
        count, bin_edges = np.histogram(averages, int(np.sqrt(value_count)))
        count = np.array(count).astype(np.float64) / value_count / (bin_edges[1] - bin_edges[0])
        bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2

        # Find two most prominent peaks in histogram (empty and occupied)
        peaks, properties = find_peaks(count, prominence=0.01)

        peak_index_in_peaks = np.argmax(properties['prominences'])
        first_peak_index = peaks[peak_index_in_peaks]
        properties['prominences'][peak_index_in_peaks] = 0
        peak_index_in_peaks = np.argmax(properties['prominences'])
        second_peak_index = peaks[peak_index_in_peaks]

        # Height of peak gives good estimate for scale
        gaussian_peak_default = 0.3989422804

        popt, _ = curve_fit(two_gaussians, bin_centers, count, p0 = (bin_centers[first_peak_index], gaussian_peak_default / count[first_peak_index], 0.5, \
            bin_centers[second_peak_index], gaussian_peak_default / count[second_peak_index], 0.5))

        # Take all sites where chance of being empty is below threshold
        threshold = norm.isf(0.001, popt[0], popt[1])
        if threshold < bin_centers[first_peak_index] or threshold > bin_centers[second_peak_index]:
            threshold = (bin_centers[first_peak_index] + bin_centers[second_peak_index]) / 2

        # Add (shifted and scaled) image detail to psf 
        psf_radius = np.linalg.norm(np.array(self.atom_locations[0]) - np.array(self.atom_locations[1])) // 2
        psf_size = int((2 * psf_radius + 1) * self.psf_supersample)
        self.psf = np.zeros((psf_size,psf_size))
        for image in images:
            if isinstance(image, DataFrame):
                image_np = image.to_numpy(np.float64)
            else:
                image_np = np.array(image).astype(np.float64)
            image_np -= average_closed_shutter_image
            for atom_location, mask in atom_location_masks:
                if np.average(image_np[mask]) > threshold:
                    padding = [[0,0],[0,0]]
                    y_min = int(atom_location[0] - psf_radius)
                    if y_min < 0:
                        padding[0][0] = -y_min
                        y_min = 0
                    y_max = int(atom_location[0] + psf_radius + 2)
                    if y_max > image_np.shape[0]:
                        padding[0][1] = y_max - image_np.shape[0]
                        y_max = image_np.shape[0]
                    x_min = int(atom_location[1] - psf_radius)
                    if x_min < 0:
                        padding[1][0] = -x_min
                        x_min = 0
                    x_max = int(atom_location[1] + psf_radius + 2)
                    if x_max > image_np.shape[1]:
                        padding[1][1] = x_max - image_np.shape[1]
                        x_max = image_np.shape[1]
                    image_detail = image_np[y_min:y_max,x_min:x_max]
                    image_detail = np.pad(image_detail, padding, mode='constant')
                    image_detail = shift(image_detail, np.floor(atom_location) - np.array(atom_location), order=1)
                    image_detail = zoom(image_detail[:-1,:-1], self.psf_supersample, order=1)
                    self.psf += image_detail
        self.psf = self.psf / np.max(self.psf)

    def calibrate(self, images, average_closed_shutter_image, angle_guesses : tuple[float,float] = (90,0), proj_shape : tuple[int,int] = (61, 61)):
        first = True
        for image in images:
            if isinstance(image, DataFrame):
                image_np = image.to_numpy(np.float64)
            else:
                image_np = np.array(image).astype(np.float64)
            if first:
                first = False
                average_filled_image = image_np
            else:
                average_filled_image += image_np

        average_filled_image /= len(images)

        # Subtract closed-shutter image to reduce pixel and row noise and clamp image at 0
        average_filled_image -= average_closed_shutter_image
        average_filled_image[average_filled_image < 0] = 0

        self._find_atom_locations(average_filled_image, angle_guesses)
        self._find_psf(images, average_closed_shutter_image)

        plt.imshow(self.psf)
        plt.show()

        trafo_site_to_image = AffineTrafo2d()
        # Set site unit vectors within image coordinate system
        trafo_site_to_image.set_origin_axes(
            magnification=self.spacing,
            angle=np.deg2rad(self.angle)
        )
        trafo_site_to_image.set_offset_by_point_pair(
            [0,0], self._image_ref
        )

        ipsf_gen = state_reconstruction.IntegratedPsfGenerator(
            psf=self.psf, psf_supersample=self.psf_supersample
        )

        proj_gen = state_reconstruction.ProjectorGenerator(
            trafo_site_to_image=trafo_site_to_image,
            integrated_psf_generator=ipsf_gen,
            proj_shape=proj_shape
        )

        # Pre-calculate projectors (this may take up to a few minutes)
        proj_gen.setup_cache(print_progress=True)

        self.solver = neutral_atom_image_analysis_cpp.ImageAnalysisProjection(self.psf, self.atom_locations)
        self.solver.setProjGen(proj_gen)

    def reconstruct(self, image):
        # Preprocess image
        image = np.array(image, dtype=float)
        if np.isfortran(image):
            image = np.ascontiguousarray(image)
        
        parameters = self.solver.reconstruct(image)
        return parameters