"""
Neutral atom image analysis wrapper
"""

import sys
import os
if sys.platform == "win32":
    os.add_dll_directory(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__) + "/")
from datetime import datetime
import math
import neutral_atom_image_analysis_cpp
import matplotlib.pyplot as plt
from pandas import DataFrame
from skimage.transform import radon
from scipy.interpolate import interp1d
from scipy.ndimage import shift
from scipy.optimize import curve_fit, OptimizeWarning, fsolve
from scipy.stats import norm, skewnorm
from scipy.signal import find_peaks, convolve2d
from scipy.special import gamma, gammaincc
import numpy as np
import abc

gaussian_peak_default = 0.3989422804

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
    def __init__(self, psf_supersample = 1, print_info = False):
        ImageAnalysis.__init__(self)
        self.psf_supersample = psf_supersample
        self.print_info = print_info

    def __three_gaussian_peaks(self, x, loc, scale, o_scale1, o_scale2, o_scale3, offset, slope):
        return offset + slope * x + norm.pdf(x, loc = loc, scale = scale) * o_scale1 + \
            norm.pdf(x, loc = loc * 2, scale = scale) * o_scale2 + \
            norm.pdf(x, loc = loc * 3, scale = scale) * o_scale3

    def __two_gaussians(self, x, loc1, scale1, f, loc2, scale2):
        return norm.pdf(x, loc = loc1, scale = scale1) * (1 - f) + \
            norm.pdf(x, loc = loc2, scale = scale2) * f

    def __two_skewed_gaussians(self, x, scew1, loc1, scale1, f, scew2, loc2, scale2):
        return skewnorm.pdf(x, scew1, loc = loc1, scale = scale1) * (1 - f) + \
            skewnorm.pdf(x, scew2, loc = loc2, scale = scale2) * f
    
    def __normalized_super_asymmetric_gaussian(self, x, loc, scale, exp_a, exp_b):
        return ((x <= loc) * np.exp(-abs(x - loc) ** exp_a / scale ** 2) + \
                (x > loc) * np.exp(-abs(x - loc) ** exp_b / scale ** 2)) /\
                (scale ** (2 / exp_a) * gamma(1 / exp_a) / exp_a + scale ** (2 / exp_b) * gamma(1 / exp_b) / exp_b)
    
    def __cumulative_normalized_super_asymmetric_gaussian(self, x, loc, scale, exp_a, exp_b):
        if x < loc:
            return (scale ** (2 / exp_a) * gamma(1 / exp_a) * gammaincc(1 / exp_a, (loc - x) ** exp_a / scale ** 2) / exp_a) /\
                (scale ** (2 / exp_a) * gamma(1 / exp_a) / exp_a + scale ** (2 / exp_b) * gamma(1 / exp_b) / exp_b)
        else:
            return 1 - (scale ** (2 / exp_b) * gamma(1 / exp_b) * gammaincc(1 / exp_b, (x - loc) ** exp_b / scale ** 2) / exp_b) /\
                (scale ** (2 / exp_a) * gamma(1 / exp_a) / exp_a + scale ** (2 / exp_b) * gamma(1 / exp_b) / exp_b)

    def __gaussian_plus_super_asymmetric_gaussian(self, x, loc1, scale1, f, loc2, scale2, exp2_a, exp2_b):
        return norm.pdf(x, loc = loc1, scale = scale1) * (1 - f) + \
            self.__normalized_super_asymmetric_gaussian(x, loc2, scale2, exp2_a, exp2_b) * f

    def __gaussian_minus_super_asymmetric_gaussian(self, x, loc1, scale1, f, loc2, scale2, exp2_a, exp2_b):
        return norm.pdf(x, loc = loc1, scale = scale1) * (1 - f) - \
            self.__normalized_super_asymmetric_gaussian(x, loc2, scale2, exp2_a, exp2_b) * f

    def __gaussian_peak_empty(self, x, loc1, scale1, f):
        return norm.pdf(x, loc = loc1, scale = scale1) * (1 - f)

    def __single_sloped_gaussian_peak(self, x, loc, scale, o_scale1, offset, slope):
        return offset + slope * x + norm.pdf(x, loc = loc, scale = scale) * o_scale1

    def __gaussian_2d(self, x, loc_x, loc_y, scale_x, scale_y, offset, mult):
        return np.array(norm.pdf(x[0], loc = loc_y, scale = scale_y) * norm.pdf(x[1], loc = loc_x, scale = scale_x) * mult + offset).ravel()
    
    def __gaussian_2d_unravelled(self, x, loc_x, loc_y, scale_x, scale_y, offset, mult):
        return np.array(norm.pdf(x[0], loc = loc_y, scale = scale_y) * norm.pdf(x[1], loc = loc_x, scale = scale_x) * mult + offset)
    
    def __gaussian_wrapped(self, x, loc, scale, mult):
        return norm.pdf(x, loc=loc, scale=scale) * mult

    def _find_best_projection_angle(self, image, center_angle, angle_radius, angle_steps, closed_shutter_image_provided):
        tested_angles = np.linspace(center_angle - angle_radius, center_angle + angle_radius, angle_steps)
        h = radon(image, theta=tested_angles, preserve_range=True)
        h_var_diff = np.diff(np.var(h, axis=0))
        h_var_second_diff = np.diff(h_var_diff)
        peak_locs, data = find_peaks(-h_var_second_diff, height=0)
        peak_heights = data['peak_heights']
        highest_var_index = peak_locs[np.argmax(peak_heights)] + 1
        peak_heights[np.argmax(peak_heights)] = np.min(peak_heights)
        second_highest_peak = np.max(peak_heights)
        second_highest_var_index = peak_locs[np.argmax(peak_heights)] + 1
        peak_heights[np.argmax(peak_heights)] = np.min(peak_heights)
        third_highest_peak = np.max(peak_heights)
        # Filter out artifacts at 0 and 90 degrees due to camera noise
        if (abs(tested_angles[highest_var_index] % 90) < 0.1 or abs(tested_angles[highest_var_index] % 90 - 90) < 0.1)\
            and second_highest_peak >= 3 * third_highest_peak and not closed_shutter_image_provided:
            highest_var_index = second_highest_var_index
        for val in h_var_diff[highest_var_index:]:
            if val > 0:
                highest_var_index += 1
            else:
                break
        return tested_angles[highest_var_index], h[...,highest_var_index].flatten()

    def _find_atom_locations(self, average_image, site_detection_threshold, extend_locations_to_fov, 
                             closed_shutter_image_provided, target_axes, optimize_locations_individually,
                             remove_sites_under_fit_height_percentile):
        origins_first_peak_axes = []
        dirs_first_peak_axes = []

        # First x, then y, angle with respect to target axis
        self.angle = [0,0]
        self.spacing = [0,0]
        self.sites_shape = [0,0]
        shape = average_image.shape

        if shape[0] < shape[1]:
            before_padding = (shape[1] - shape[0]) // 2
            padding = ((before_padding, (shape[1] - shape[0]) - before_padding), (0,0))
        elif shape[1] < shape[0]:
            before_padding = (shape[0] - shape[1]) // 2
            padding = ((0,0), (before_padding, (shape[0] - shape[1]) - before_padding))
        else:
            padding = ((0,0),(0,0))
        used_average_image = np.pad(average_image, padding, mode='constant', constant_values=np.median(average_image))

        # Handle both dimensions separately starting from a rough guess of the angle
        for dim in range(2):
            # Find the best suited angle within +- 15 degrees
            best_angle, _ = self._find_best_projection_angle(used_average_image, target_axes[dim], 15, 101, closed_shutter_image_provided)

            # Find more precise angle within +- 1 degree of approx angle
            best_angle, projection = self._find_best_projection_angle(used_average_image, best_angle, 2, 101, closed_shutter_image_provided)

            if self.print_info:
                print("Angle of axis determined to be " + str(best_angle))
            self.angle[dim] = target_axes[dim] - best_angle

            # Use existing radon transform to acquire image projection at given angle
            projection = projection[padding[dim][0]:projection.shape[0]-padding[dim][1]]
            projection -= (projection[0] + projection[-1]) / 2

            # Compute autocorrelation to find periodicity
            autocorrelation = np.correlate(projection, projection, mode='full')
            autocorrelation = autocorrelation[autocorrelation.size//2:]
            if self.print_info:
                plt.plot(autocorrelation)
                plt.title("Autocorrelation")
                plt.show()
            index = 0
            for i, v in enumerate(np.diff(autocorrelation)):
                if v > 0:
                    index = i
                    break

            # Find first peak (other than x=0) and fit three gaussians to the first three peaks (to improve precision)
            index += np.argmax(autocorrelation[index:])
            start_index = index // 2
            end_index = int(3.5 * index) + 1
            if end_index > len(autocorrelation):
                end_index = len(autocorrelation)
            peak_width_guess = index / 5
            peak_factor_guess = (autocorrelation[index] - autocorrelation[start_index]) / norm.pdf([0],0,peak_width_guess)[0]
            x_range = range(start_index,end_index)
            slope = (autocorrelation[start_index + index] - autocorrelation[start_index]) / index
            popt = None
            try:
                popt, _ = curve_fit(self.__three_gaussian_peaks, x_range, autocorrelation[start_index:end_index], 
                    p0=[index, peak_width_guess, peak_factor_guess, peak_factor_guess, peak_factor_guess, autocorrelation[start_index], slope],
                    bounds=([0, 0, 0, 0, 0, np.min(autocorrelation[start_index:end_index]), -np.inf], 
                            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
            except (RuntimeError, OptimizeWarning):
                if popt is None:
                    try:
                        popt, _ = curve_fit(self.__single_sloped_gaussian_peak, x_range, autocorrelation[start_index:end_index], 
                            p0=[index, peak_width_guess, peak_factor_guess, autocorrelation[start_index], slope],
                            bounds=([0, 0, 0, np.min(autocorrelation[start_index:end_index]), -np.inf], 
                                    [np.inf, np.inf, np.inf, np.inf, np.inf]))
                    except (RuntimeError, OptimizeWarning):
                        print("All curve fitting for atom site detection failed. Using rough estimate")
                        popt = [index, peak_width_guess]
            self.spacing[dim] = popt[0]
            peak_width = popt[1]

            if self.print_info:
                print("Spacing " + str(popt[0]))

            # Acquire rolling sum over five adjacent elements of projection to smooth over noise
            neighbor_dist = 2
            if self.spacing[dim] < 4:
                neighbor_dist = 1
            if self.spacing[dim] < 2:
                neighbor_dist = 0
            padded_n_f = np.pad(projection, (neighbor_dist,neighbor_dist), 'constant')
            neighboring_series = padded_n_f[neighbor_dist:padded_n_f.shape[0] - neighbor_dist]
            for d in range(neighbor_dist):
                neighboring_series += padded_n_f[d:-2 * neighbor_dist + d] + \
                    padded_n_f[2 * neighbor_dist - d:padded_n_f.shape[0] - d]

            # Use smoothed projection to find offset for which the values at the periodic locations are maximized on average
            max_average_value = None
            interpolation = interp1d(np.arange(len(neighboring_series)), neighboring_series)
            for offset in np.linspace(0, self.spacing[dim], 100, endpoint=False):
                index_count = int((len(neighboring_series) - 1 - offset) / self.spacing[dim]) + 1
                indices = np.array(range(index_count)) * self.spacing[dim] + offset
                values = interpolation(indices)
                average_value = np.average(values)
                if max_average_value is None or average_value > max_average_value:
                    max_average_value = average_value
                    maximizing_values = values
                    maximizing_indices = indices

            # At the given periodic locations, take all that are at least site_detection_threshold times the maximum 
            # or that lie in the middle of locations where that is the case
            # Allow sites on the edge to be only above half threshold if they are the maximum or minimum of diff of values
            threshold = maximizing_values.max() * site_detection_threshold
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
            peak_height = 1 / norm.pdf([0],0,peak_width)[0] / (2 * neighbor_dist + 1)
            def gaussian_series(x, loc_offset, scale, factor):
                result = 0
                for index, value in zip(indices, values):
                    result += norm.pdf(x, loc = index + loc_offset, scale = scale) * value * factor
                return result

            x_range = [i for i in range(len(projection)) if projection[i] >= 0]
            try:
                popt, _ = curve_fit(gaussian_series, x_range, projection[x_range], p0 = [0,peak_width,peak_height])
                for i in range(len(indices)):
                    indices[i] += popt[0]
            except (RuntimeError, ValueError, OptimizeWarning):
                print("Precise subpixel locations could not be established due to curve fit error")
            
            if self.print_info:
                plt.plot(x_range, projection[x_range])
                plt.plot(x_range, gaussian_series(x_range,*popt))
                plt.title("Fitting multiple gaussian peaks to emission projection")
                plt.legend(["Proj", "Fit"])
                plt.show()

            # Save origin and direction of axis along first row/column of atom sites
            # Saved direction is along projection axis, meaning it would be projected onto a point
            origin_loc = np.array((used_average_image.shape[0] // 2, used_average_image.shape[1] // 2)).astype(np.float64)
            proj_vector = np.array((-np.sin(np.deg2rad(best_angle)), np.cos(np.deg2rad(best_angle))))
            origin_loc += (indices[0] - float(len(projection) // 2)) * proj_vector
            origin_loc -= np.array([padding[0][0],padding[1][0]])
            origins_first_peak_axes.append(origin_loc)
            direction = np.array([np.cos(np.deg2rad(best_angle)), np.sin(np.deg2rad(best_angle))])
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
        if extend_locations_to_fov:
            dirs_and_start_dists = [(-1,1),(1,0)]
            for r_dir, r_start_dist in dirs_and_start_dists:
                row_in_fov = True
                r = r_start_dist
                while row_in_fov:
                    row_in_fov = False
                    row_point = self._image_ref + r_dir * r * self.spacing[0] * proj_vectors[0]
                    for c_dir, c_start_dist in dirs_and_start_dists:
                        location_in_fov = True
                        c = c_start_dist
                        while location_in_fov:
                            location = row_point + c_dir * c * self.spacing[1] * proj_vectors[1]
                            if location[0] >= 0 and location[1] >= 0 and location[0] < average_image.shape[0] and location[1] < average_image.shape[1]:
                                self.atom_locations.append(location)
                                row_in_fov = True
                            else:
                                location_in_fov = False
                            c += 1
                    r += 1
        else:
            for r in range(self.sites_shape[0]):
                for c in range(self.sites_shape[1]):
                    location = self._image_ref + r * self.spacing[0] * proj_vectors[0] + c * self.spacing[1] * proj_vectors[1]
                    if min(location) >= 0 and location[0] < average_image.shape[0] and location[1] < average_image.shape[1]:
                        self.atom_locations.append(location)
        
        if optimize_locations_individually:
            all_mults = []
            for i in range(len(self.atom_locations)):
                x, y = self.atom_locations[i]
                y_start = int(y - self.spacing[0] / 2)
                y_max = int(self.spacing[0])
                if y_start < 0:
                    y_start = 0
                if y_start + y_max > average_image.shape[0]:
                    y_max = average_image.shape[0] - y_start
                x_start = int(x - self.spacing[1] / 2)
                x_max = int(self.spacing[1])
                if x_start < 0:
                    x_start = 0
                if x_start + x_max > average_image.shape[1]:
                    x_max = average_image.shape[1] - x_start
                x_data = np.mgrid[0:x_max - 2, 0:y_max - 2]
                y_data = np.array(average_image[x_start:x_start + x_max, y_start:y_start + y_max])
                max_3x3 = convolve2d(np.array(y_data), \
                    self.__gaussian_2d_unravelled(np.mgrid[0:3, 0:3], 1, 1, 1, 1, 0, 1), mode='valid')
                max_index = np.unravel_index(max_3x3.argmax(), max_3x3.shape)
                init_guesses = [max_index[1] + 1, max_index[0] + 1, 5, 5, max_3x3.min(),\
                                (max_3x3.max() - max_3x3.min()) * gaussian_peak_default * gaussian_peak_default * 25]
                all_bounds = ([0,0,0,0,-np.inf,0],[y_max,x_max,np.inf,np.inf,np.inf,np.inf])
                try:
                    popt, _ = curve_fit(self.__gaussian_2d, x_data, max_3x3.ravel(), p0=init_guesses, bounds=all_bounds)
                except: 
                    all_mults.append(0)
                    continue
                all_mults.append(popt[5])
                self.atom_locations[i] = np.array((x_start + popt[1], y_start + popt[0]))
        elif remove_sites_under_fit_height_percentile is not None and remove_sites_under_fit_height_percentile > 0:
            all_mults = []
            for i in range(len(self.atom_locations)):
                x, y = self.atom_locations[i]
                y_start = int(y - self.spacing[0] / 2)
                y_max = int(self.spacing[0])
                if y_start < 0:
                    y_start = 0
                if y_start + y_max > average_image.shape[0]:
                    y_max = average_image.shape[0] - y_start
                x_start = int(x - self.spacing[1] / 2)
                x_max = int(self.spacing[1])
                if x_start < 0:
                    x_start = 0
                if x_start + x_max > average_image.shape[1]:
                    x_max = average_image.shape[1] - x_start
                x_data = np.mgrid[0:x_max, 0:y_max]
                y_data = np.array(average_image[x_start:x_start + x_max, y_start:y_start + y_max])
                max_3x3 = convolve2d(np.array(y_data), np.ones((3,3)), mode='valid')
                max_index = np.unravel_index(max_3x3.argmax(), max_3x3.shape)
                init_guesses = [5, 5, y_data.min(), (y_data.max() - y_data.min()) * gaussian_peak_default * gaussian_peak_default * 25]
                all_bounds = ([0,0,-np.inf,0],[np.inf,np.inf,np.inf,np.inf])
                try:
                    popt, _ = curve_fit(lambda data, scale_x, scale_y, offset, mult: \
                                        self.__gaussian_2d(data, y - y_start, x - x_start, scale_x, scale_y, offset, mult), \
                                        x_data, y_data.ravel(), p0=init_guesses, bounds=all_bounds)
                except:
                    all_mults.append(0)
                    continue
                all_mults.append(popt[3])

        if remove_sites_under_fit_height_percentile is not None and remove_sites_under_fit_height_percentile > 0:
            bin_count = int(math.sqrt(len(self.atom_locations)))
            if bin_count < 10:
                bin_count = 10
            hist, bin_edges = np.histogram(all_mults, bin_count)
            x_data = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2
            hist = hist.astype(float) / len(all_mults)
            stddev_guess = np.std(all_mults)
            try:
                popt, _ = curve_fit(self.__gaussian_wrapped, x_data, hist, p0=[x_data[int(len(x_data) * 0.75)],stddev_guess,bin_edges[1]-bin_edges[0]])
            except:
                if self.print_info:
                    print("Curve fit for site removal not successful")
            else:
                new_atom_locations = []
                site_removal_threshold = norm.ppf(remove_sites_under_fit_height_percentile, loc=popt[0], scale=popt[1])
                if self.print_info:
                    print("Removing atom site where chance of peak height is below " + str(remove_sites_under_fit_height_percentile) + " if atoms were present")
                for i, mult in enumerate(all_mults):
                    if mult >= site_removal_threshold:
                        new_atom_locations.append(self.atom_locations[i])
                self.atom_locations = new_atom_locations

    
    def _get_potentially_occ_atom_locations_within_radius(self, center, atom_location_occ_groups, radius):
        ret_locations = []
        center_np = np.array(center)
        for atom_location, occupancy in atom_location_occ_groups:
            if occupancy != 0:
                dist = np.linalg.norm(center_np - np.array(atom_location))
                if dist > 1e-5 and dist <= radius:
                    ret_locations.append(atom_location)
        return ret_locations
    
    def _coord_is_closer_than_other(self, center, location, other_locations):
        target_dist = np.linalg.norm(center - location)
        for atom_location in other_locations:
            dist = np.linalg.norm(atom_location - location)
            if dist < target_dist:
                return False
        return True
    
    def _get_atom_sites_in_subshape(self, atom_locations, occ_count, shape_start, shape_end, overlap):
        all_local_locations = []
        local_to_all_index_mapping = []
        for i, atom_location in enumerate(atom_locations):
            if atom_location[0] >= shape_start[0] - overlap and atom_location[0] < shape_end[0] + overlap and \
                atom_location[1] >= shape_start[1] - overlap and atom_location[1] < shape_end[1] + overlap:
                if i < occ_count:
                    local_to_all_index_mapping.append(i)
                else:
                    local_to_all_index_mapping.append(-1)
                all_local_locations.append(atom_location)

        return all_local_locations, local_to_all_index_mapping
    
    def _find_psf(self, images, average_closed_shutter_image, psf_distance_mult):
        if self.atom_locations is None:
            raise AttributeError("Atom locations not yet set", name="atom_locations", obj=self)
        half_min_spacing = min(self.spacing) / 2
        roi_radius = min(half_min_spacing, 5)
        psf_radius = int(4 * half_min_spacing)
        unsupersized_psf_size = int((2 * psf_radius + 1))
        psf_size = int(unsupersized_psf_size * self.psf_supersample)
        self.psf = np.zeros((psf_size,psf_size),float)
        psf_per_pixel_count = np.zeros_like(self.psf, int)
        
        # Integrate over roi around atom sites to determine sites to use for PSF generation
        atom_location_masks = []

        for atom_location in self.atom_locations:
            y_start = int(atom_location[0] - roi_radius)
            if y_start < 0:
                y_start = 0
            y_end = int(atom_location[0] + roi_radius + 2)
            if y_end > average_closed_shutter_image.shape[0]:
                y_end = average_closed_shutter_image.shape[0]
            x_start = int(atom_location[1] - roi_radius)
            if x_start < 0:
                x_start = 0
            x_end = int(atom_location[1] + roi_radius + 2)
            if x_end > average_closed_shutter_image.shape[1]:
                x_end = average_closed_shutter_image.shape[1]
            Y, X = np.ogrid[y_start:y_end,x_start:x_end]
            dist_from_center = np.sqrt((Y - atom_location[0])**2 + (X - atom_location[1])**2)
            mask = dist_from_center <= roi_radius
            atom_location_masks.append((atom_location, mask))

        averages = []
        averages_by_image_index = []
        for image in images:
            local_average_list = []
            if isinstance(image, DataFrame):
                image_np = image.to_numpy(np.float64)
            else:
                image_np = np.array(image).astype(np.float64)
            image_np -= average_closed_shutter_image
            for atom_location, mask in atom_location_masks:
                y_start = int(atom_location[0] - roi_radius)
                if y_start < 0:
                    y_start = 0
                y_end = int(atom_location[0] + roi_radius + 2)
                if y_end > image_np.shape[0]:
                    y_end = image_np.shape[0]
                x_start = int(atom_location[1] - roi_radius)
                if x_start < 0:
                    x_start = 0
                x_end = int(atom_location[1] + roi_radius + 2)
                if x_end > image_np.shape[1]:
                    x_end = image_np.shape[1]
                
                background_dist = 10
                by_start = y_start - background_dist
                if by_start < 0:
                    by_start = 0
                by_end = y_end + background_dist
                if by_end > image_np.shape[0]:
                    by_end = image_np.shape[0]
                bx_start = x_start - background_dist
                if bx_start < 0:
                    bx_start = 0
                bx_end = x_end + background_dist
                if bx_end > image_np.shape[1]:
                    bx_end = image_np.shape[1]
                image_detail = image_np[y_start:y_end,x_start:x_end]
                fill_value = np.min(image_detail)
                image_detail[np.invert(mask)] = fill_value
                peak_coords = np.unravel_index(np.argmax(image_detail), image_detail.shape) + np.array([y_start,x_start])
                #peak_coords = (np.round(atom_location)).astype(int)

                trough_brightness = []
                y_start = int(np.round(atom_location[0] - self.spacing[0] / 2))
                y_end = int(np.round(atom_location[0] + self.spacing[0] / 2))
                x_start = int(np.round(atom_location[1] - self.spacing[1] / 2))
                x_end = int(np.round(atom_location[1] + self.spacing[1] / 2))

                t_start = x_start
                if t_start < 0:
                    t_start = 0
                t_end = x_end
                if t_end > image_np.shape[1]:
                    t_end = image_np.shape[1]
                if t_end > t_start:
                    if y_start >= 0:
                        trough_brightness.extend(image_np[y_start, t_start:t_end])
                    if y_end < image_np.shape[0]:
                        trough_brightness.extend(image_np[y_end, t_start:t_end])

                # Add and subtract 1 since corners are already accounted for
                t_start = y_start + 1
                if t_start < 0:
                    t_start = 0
                t_end = y_end - 1
                if t_end > image_np.shape[0]:
                    t_end = image_np.shape[0]
                if t_end > t_start:
                    if x_start >= 0:
                        trough_brightness.extend(image_np[t_start:t_end, x_start])
                    if x_end < image_np.shape[1]:
                        trough_brightness.extend(image_np[t_start:t_end, x_end])

                if peak_coords[0] >= 0 and peak_coords[0] < image_np.shape[0] and peak_coords[1] >= 0 and peak_coords[1] < image_np.shape[1]:
                    avg = np.average(image_detail[mask])
                    averages.append(avg)
                    local_average_list.append((atom_location,avg))
            averages_by_image_index.append(local_average_list)

        value_count = len(images) * len(self.atom_locations)
        count, bin_edges = np.histogram(averages, int(np.sqrt(value_count)))
        count = np.array(count).astype(np.float64) / value_count / (bin_edges[1] - bin_edges[0])
        bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2

        # Find two most prominent peaks in histogram (empty and occupied)
        peaks, properties = find_peaks(count, prominence=0.01)

        # Height of peak gives good estimate for scale
        popt = None
        peak_index_in_peaks = np.argmax(properties['prominences'])
        first_peak_index = peaks[peak_index_in_peaks]
        if len(peaks) > 1:
            properties['prominences'][peak_index_in_peaks] = 0
            peak_index_in_peaks = np.argmax(properties['prominences'])
            second_peak_index = peaks[peak_index_in_peaks]

            try:
                popt, _ = curve_fit(self.__two_gaussians, bin_centers, count, p0 = (bin_centers[first_peak_index], gaussian_peak_default / count[first_peak_index], 0.5, \
                    bin_centers[second_peak_index], gaussian_peak_default / count[second_peak_index]))
                # Take all sites where chance of being empty is below threshold
                pdf_empty = norm.pdf(bin_centers, loc=popt[0], scale=popt[1]) * (1 - popt[2])
                pdf_occ = norm.pdf(bin_centers, loc=popt[3], scale=popt[4]) * popt[2]
                empty_threshold = -1
                threshold = -1
                mask = pdf_empty > 0
                max_occ_to_empty_prob = np.max(pdf_occ[mask] / pdf_empty[mask])
                if max_occ_to_empty_prob > 1:
                    for i in range(first_peak_index, len(count)):
                        if pdf_occ[i] > pdf_empty[i]:
                            empty_threshold = bin_centers[i]
                            break
                    threshold_prob = 1e3
                    if threshold_prob > max_occ_to_empty_prob:
                        threshold_prob = (max_occ_to_empty_prob + 1) / 2
                    for i in range(first_peak_index, len(count)):
                        if pdf_occ[i] / pdf_empty[i] >= threshold_prob:
                            threshold = bin_centers[i]
                            break
                if threshold < bin_centers[first_peak_index]:
                    threshold = (bin_centers[first_peak_index] + bin_centers[second_peak_index]) / 2
                if empty_threshold > bin_centers[second_peak_index]:
                    empty_threshold = bin_centers[first_peak_index]
            except (RuntimeError, ValueError, OptimizeWarning):
                if popt is None or threshold is None or empty_threshold is None or np.isnan(threshold) or np.isnan(empty_threshold):
                    print("Curve fitting for threshold for psf acquisition failed. Using rough estimate")
                    threshold = (bin_centers[first_peak_index] + bin_centers[second_peak_index]) / 2
                    empty_threshold = bin_centers[first_peak_index]
        else:
            try:
                popt, _ = curve_fit(self.__gaussian_peak_empty, bin_centers, count, p0 = (bin_centers[first_peak_index], gaussian_peak_default / count[first_peak_index], 0.5))
                # Take all sites where chance of being empty is below threshold
                threshold = norm.isf(1e-3 / (1 - popt[2]), popt[0], popt[1])
                empty_threshold = norm.isf(0.1 / (1 - popt[2]), popt[0], popt[0])
            except (RuntimeError, ValueError, OptimizeWarning):
                if popt is None or threshold is None or empty_threshold is None or np.isnan(threshold) or np.isnan(empty_threshold):
                    print("Curve fitting for threshold for psf acquisition failed. Using rough estimate")
                    threshold = (bin_centers[first_peak_index] + len(count)) / 2
                    empty_threshold = bin_centers[first_peak_index]

        voronoi_generator_cpp = neutral_atom_image_analysis_cpp.VoronoiGenerator()

        for image_index,image in enumerate(images):
            if isinstance(image, DataFrame):
                image_np = image.to_numpy(np.float64)
            else:
                image_np = np.array(image).astype(np.float64)
            image_np -= average_closed_shutter_image

            # For every atom location, save rough occupancy
            occupied_atom_locations = []
            potentially_occupied_atom_locations = []
            for atom_location, avg in averages_by_image_index[image_index]:
                if avg < empty_threshold:
                    if avg > threshold:
                        potentially_occupied_atom_locations.append(atom_location)
                else:
                    if avg <= threshold:
                        potentially_occupied_atom_locations.append(atom_location)
                    else:
                        occupied_atom_locations.append(atom_location)

            padding = np.array(((psf_radius,psf_radius),(psf_radius,psf_radius)))
            image_np = np.pad(image_np, padding, mode='constant')
            complete_voronoi = np.full_like(image_np, -1)

            voronoi_generator_cpp.generate(image_np, complete_voronoi, occupied_atom_locations, \
                                           potentially_occupied_atom_locations, psf_distance_mult, padding[0][0], padding[0][1])

            mask = np.ones_like(self.psf, bool)
            for i, atom_location in enumerate(occupied_atom_locations):
                y_padding = 0
                y_min = int(np.floor(atom_location[0] - psf_radius)) + padding[0][0]
                y_max = int(np.floor(atom_location[0] + psf_radius + 2)) + padding[0][0]
                if y_max > image_np.shape[0]:
                    y_padding = y_max - image_np.shape[0]
                    y_max = image_np.shape[0]
                x_padding = 0
                x_min = int(np.floor(atom_location[1] - psf_radius)) + padding[1][0]
                x_max = int(np.floor(atom_location[1] + psf_radius + 2)) + padding[1][0]
                if x_max > image_np.shape[1]:
                    x_padding = x_max - image_np.shape[1]
                    x_max = image_np.shape[1]
                image_detail = image_np[y_min:y_max,x_min:x_max]

                if x_padding > 0 or y_padding > 0:
                    image_detail = np.pad(image_detail, ((0, y_padding),(0, x_padding)), mode='constant')
                image_detail = np.kron(image_detail, np.ones((self.psf_supersample,self.psf_supersample)))
                shift_amount = self.psf_supersample * (np.floor(atom_location) - np.array(atom_location))
                image_detail = shift(image_detail, shift_amount, order=1)
                image_detail = image_detail[:-self.psf_supersample,:-self.psf_supersample]
                mask = complete_voronoi[y_min:y_max,x_min:x_max] == i
                
                if np.sum(np.sum(mask,axis=0)>0) + np.sum(np.sum(mask,axis=1)>0) > self.spacing[0] + self.spacing[1]:
                    if x_padding > 0 or y_padding > 0:
                        mask = np.pad(mask, ((0, y_padding),(0, x_padding)), mode='constant', constant_values=False)
                    mask = np.kron(mask.astype(float), np.ones((self.psf_supersample,self.psf_supersample))).astype(bool)
                    mask &= shift(mask, np.sign(shift_amount), order=1)
                    mask = mask[:-self.psf_supersample,:-self.psf_supersample]
                    image_detail[np.invert(mask)] = 0

                    psf_per_pixel_count += mask.astype(int)
                    self.psf += image_detail
        
        while 0 in psf_per_pixel_count:
            self.psf = self.psf[1:-1,1:-1]
            psf_per_pixel_count = psf_per_pixel_count[1:-1,1:-1]

        for y, x in np.ndindex(self.psf.shape):
            if not (psf_per_pixel_count[y,x] == 0 or np.isnan(self.psf[y,x])):
                self.psf[y,x] = self.psf[y,x] / psf_per_pixel_count[y,x]
        self.psf -= np.min(self.psf)
        for y, x in np.ndindex(self.psf.shape):
            if psf_per_pixel_count[y,x] == 0 or np.isnan(self.psf[y,x]):
                self.psf[y,x] = 0
        self.psf = self.psf / np.max(self.psf)

    def _get_average_images(self, images, camera_noise_reduction_method, average_closed_shutter_image):
        border_pixels = []
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
            border_pixels.extend(image_np[0,...])
            border_pixels.extend(image_np[-1,...])
            border_pixels.extend(image_np[1:-1,0])
            border_pixels.extend(image_np[1:-1,-1])

        average_filled_image /= len(images)

        closed_shutter_image_provided = average_closed_shutter_image is not None
        if camera_noise_reduction_method == "border":
            background_brightness = np.median(border_pixels)
            average_closed_shutter_image = np.full_like(image_np, background_brightness)
        # Use row / column median as fallback if image is specified but not provided
        elif camera_noise_reduction_method == "rowcol" or not closed_shutter_image_provided:
            average_closed_shutter_image = np.zeros_like(image_np)
            for row in range(average_filled_image.shape[0]):
                average_closed_shutter_image[row,...] += np.median(average_filled_image[row,...])
            for col in range(average_filled_image.shape[1]):
                average_closed_shutter_image[...,col] += np.median((average_filled_image - average_closed_shutter_image)[...,col])

        # Subtract closed-shutter image to reduce pixel and row noise
        average_filled_image -= average_closed_shutter_image

        return average_filled_image, average_closed_shutter_image
    

    def _generate_and_set_projectors_low_spacing(self, proj_shape):
        try:
            import state_reconstruction
            from libics.tools.trafo.linear import AffineTrafo2d
        except ImportError:
            if self.print_info:
                print("Projectors should have been generated using embedded neighboring PSFs, but requirements state_reconstruction and/or libics are not installed")
            self._generate_and_set_projectors_no_dependencies(proj_shape)
        else:
            trafo_site_to_image = AffineTrafo2d()
            # Set site unit vectors within image coordinate system
            trafo_site_to_image.set_origin_axes(
                magnification=(self.spacing[1],self.spacing[0]),
                angle=np.deg2rad((self.angle[1],self.angle[0]))
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

            if self.print_info:
                if self.psf_supersample > 1:
                    print("Integrated projector(s):")
                    fig, ax = plt.subplots(self.psf_supersample, self.psf_supersample)
                    for i in range(self.psf_supersample):
                        for j in range(self.psf_supersample):
                            fig.colorbar(ax[i,j].imshow(proj_gen.proj_cache[i,j]), ax = ax[i,j])
                    fig.show()
                else:
                    plt.imshow(proj_gen.proj_cache[0, 0])
                    plt.title("Integrated projector")
                    plt.colorbar()
                plt.show()

            # Create object in underlying C++ library and set projectors
            if self.print_info:
                print("Creating C++ object")
            self.solver.setProjectors(proj_gen)


    def _generate_and_set_projectors_no_dependencies(self, proj_shape):
        if proj_shape is None:
            proj_shape = self.psf.shape // self.psf_supersample - 2
        if proj_shape[0] <= 0 or proj_shape[1] <= 0:
            raise ValueError("Projection shape determined to be zero or negative")
        full_projectors_array = np.ndarray((self.psf_supersample, self.psf_supersample, proj_shape[0], proj_shape[1]))
        for dx in range(self.psf_supersample):
            for dy in range(self.psf_supersample):
                image_pos = np.array([dx, dy]) - self.psf_supersample // 2

                binned_psf = shift(self.psf, image_pos)
                binned_psf = binned_psf.reshape(self.psf.shape[0] // self.psf_supersample, self.psf_supersample, \
                                                self.psf.shape[1] // self.psf_supersample, self.psf_supersample).mean(axis=3).mean(axis=1)
                
                binned_psf = binned_psf.reshape((1, -1))
                binned_psf = np.linalg.pinv(binned_psf)
                binned_psf = binned_psf.reshape((self.psf.shape[0] // self.psf_supersample, self.psf.shape[1] // self.psf_supersample))

                crop = np.array(binned_psf.shape) - np.array(proj_shape)
                binned_psf = binned_psf[crop[0] // 2:-((crop[0] + 1) // 2), crop[1] // 2:-((crop[1] + 1) // 2)]

                full_projectors_array[dx,dy] = binned_psf

        if self.print_info:
            if self.psf_supersample > 1:
                print("Integrated projector(s):")
                fig, ax = plt.subplots(self.psf_supersample, self.psf_supersample)
                for i in range(self.psf_supersample):
                    for j in range(self.psf_supersample):
                        fig.colorbar(ax[i,j].imshow(full_projectors_array[i,j]), ax = ax[i,j])
                fig.show()
            else:
                plt.imshow(full_projectors_array[0, 0])
                plt.title("Integrated projector")
                plt.colorbar()
            plt.show()
        self.solver.setProjectorsFromArray(full_projectors_array)


    def _find_atom_site_groupings(self, images, min_cal_samples):
        parameters = []
        atom_site_index_to_parameter_index = []

        # Group atom sites together spatially
        if min_cal_samples is None:
            parameters.append([])
            atom_site_index_to_parameter_index = [0] * len(self.atom_locations)
        elif len(images) >= min_cal_samples:
            for i in range(len(self.atom_locations)):
                parameters.append([])
                atom_site_index_to_parameter_index.append(i)
        else:
            atom_site_cluster_size = np.ceil(min_cal_samples / len(images))
            if atom_site_cluster_size > len(self.atom_locations) // 2:
                parameters.append([])
                atom_site_index_to_parameter_index = [0] * len(self.atom_locations)
            else:
                rows_in_group = 1
                cols_in_group = 1
                row_group_size = 0
                col_group_size = 0
                while rows_in_group * cols_in_group < atom_site_cluster_size:
                    potential_new_row_size = row_group_size + self.spacing[0]
                    potential_new_col_size = col_group_size + self.spacing[1]
                    if potential_new_row_size < potential_new_col_size or \
                        (potential_new_row_size == potential_new_col_size and \
                        rows_in_group < cols_in_group):
                        if rows_in_group + 1 > self.sites_shape[0] // 2:
                            rows_in_group = self.sites_shape[0]
                            cols_in_group = int(np.ceil(atom_site_cluster_size / rows_in_group))
                            break
                        else:
                            rows_in_group += 1
                            row_group_size = potential_new_row_size
                    else:
                        if cols_in_group + 1 > self.sites_shape[1] // 2:
                            cols_in_group = self.sites_shape[1]
                            rows_in_group = int(np.ceil(atom_site_cluster_size / cols_in_group))
                            break
                        else:
                            cols_in_group += 1
                            col_group_size = potential_new_col_size
                if self.print_info:
                    print("Grouping atom sites together to get sufficient data points")
                    print("Rows in group: " + str(rows_in_group))
                    print("Cols in group: " + str(cols_in_group))

                row_groups = np.array_split(range(self.sites_shape[0]), self.sites_shape[0] // rows_in_group)
                col_groups = np.array_split(range(self.sites_shape[1]), self.sites_shape[1] // cols_in_group)
                atom_site_index_to_parameter_index = [0] * len(self.atom_locations)
                for row_group in row_groups:
                    for col_group in col_groups:
                        parameter_index = len(parameters)
                        contains_site = False
                        for r in row_group:
                            for c in col_group:
                                if r * self.sites_shape[1] + c < len(self.atom_locations):
                                    atom_site_index_to_parameter_index[r * self.sites_shape[1] + c] = parameter_index
                                    contains_site = True
                        if contains_site:
                            parameters.append([])
        return parameters, atom_site_index_to_parameter_index

    
    def _generate_and_set_projectors(self, proj_shape):
        min_spacing = int(min(self.spacing))
        if proj_shape is None:
            print("Proj_shape set to min spacing " + str(min_spacing))
            if(min_spacing % 2 == 0):
                min_spacing += 1
            proj_shape = (min_spacing, min_spacing)
        if not hasattr(self, '_image_ref'):
            self._image_ref = self.atom_locations[0]

        if (proj_shape[0] > min_spacing or proj_shape[1] > min_spacing) \
            and hasattr(self, 'spacing') and hasattr(self, 'angle'):
            self._generate_and_set_projectors_low_spacing(proj_shape)
        else:
            self._generate_and_set_projectors_no_dependencies(proj_shape)

    
    def _calibrate_threshold(self, parameters, atom_site_index_to_parameter_index, histogram_path):
        self.threshold = [0] * len(self.atom_locations)

        fidelities = []
        fidelities0 = []
        fidelities1 = []

        average_filling_ratio = 0
        first_peaks = []
        second_peaks = []

        for p_index, parameters_individual in enumerate(parameters):
            # Prepare histogram for threshold detection
            bin_count = 2 * int(math.sqrt(len(parameters_individual)))
            count, bin_edges = np.histogram(parameters_individual, bins=bin_count)
            bin_size = bin_edges[1] - bin_edges[0]
            count = np.array(count).astype(np.float64) / len(parameters_individual) / bin_size
            bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2

            first_peak_index = np.argmax(count)
            if first_peak_index < len(count) / 2:
                start_index = 0
                end_index = 2 * first_peak_index + 1
            else:
                start_index = first_peak_index - (len(count) - first_peak_index - 1)
                end_index = len(count)
            
            popt, _ = curve_fit(self.__gaussian_wrapped, bin_centers[start_index:end_index], count[start_index:end_index], p0=[bin_centers[first_peak_index], bin_size, 1])
            first_peak = popt[0]
            first_peak_scale = popt[1]
            count_without_first_peak = count - self.__gaussian_wrapped(bin_centers, *popt)
            second_peak_index = np.argmax(count_without_first_peak)
            popt, _ = curve_fit(self.__gaussian_wrapped, bin_centers, count_without_first_peak, p0=[bin_centers[second_peak_index], bin_size, 1])
            second_peak = popt[0]
            second_peak_scale = popt[1]
            if first_peak > second_peak:
                tmp_peak = second_peak
                tmp_peak_scale = second_peak_scale
                second_peak = first_peak
                second_peak_scale = first_peak_scale
                first_peak = tmp_peak
                first_peak_scale = tmp_peak_scale

            popt = None
            popt_guesses = [first_peak, first_peak_scale, 0.5, second_peak, second_peak_scale, 2, 2]
            try:
                popt, _ = curve_fit(self.__gaussian_plus_super_asymmetric_gaussian, bin_centers, count, p0 = popt_guesses, \
                    bounds=([np.array(parameters_individual).min(), 0, 0, np.array(parameters_individual).min(), 0, 0, 0],\
                            [np.array(parameters_individual).max(), np.inf, 1, np.array(parameters_individual).max(), np.inf, np.inf, np.inf]))
            except ValueError:
                print("Either ydata or xdata contained NaNs, or incompatible options were used for curve_fitting for threshold detection! Using rough estimations")
                popt = popt_guesses
            except RuntimeError:
                print("The least-squares minimization failed for curve_fitting for threshold detection! Using rough estimations")
                popt = popt_guesses
            except OptimizeWarning:
                print("The covariance of the parameters could not be estimated for curve_fitting for threshold detection!")
                if popt is None:
                    popt = popt_guesses

            first_peak = popt[0]
            filling_ratio = popt[2]            
            second_peak = popt[3]

            t = None
            all_roots = fsolve(self.__gaussian_minus_super_asymmetric_gaussian, (first_peak + second_peak) / 2, args=tuple(popt))
            for root in all_roots:
                if root > first_peak and root < second_peak:
                    t = root
                    break
            if t is None:
                target_fidelity0 = 0.001
                if len(fidelities0) > 0:
                    target_fidelity0 = np.average(fidelities0)
                print("Intersection of first and second peak could not be established at site (group) " + str(p_index) +\
                      ". Setting threshold so fidelity 0 matches other sites")
                t = norm.ppf(target_fidelity0, loc = first_peak, scale = popt[1])

            for atom_location_index, parameter_index in enumerate(atom_site_index_to_parameter_index):
                if parameter_index == p_index:
                    self.threshold[atom_location_index] = t

            fidelity0 = norm.cdf(t, loc = first_peak, scale = popt[1])
            fidelities0.append(fidelity0)
            fidelity1 = 1 - self.__cumulative_normalized_super_asymmetric_gaussian(t, popt[3], popt[4], popt[5], popt[6])
            fidelities1.append(fidelity1)
            fidelities.append((1 - filling_ratio) * fidelity0 + filling_ratio * fidelity1)

            average_filling_ratio += filling_ratio / len(parameters)
            first_peaks.append(first_peak)
            second_peaks.append(second_peak)

            if histogram_path is not None and isinstance(histogram_path, str):
                plt.plot(bin_centers, count)
                plt.plot(bin_centers, self.__gaussian_plus_super_asymmetric_gaussian(bin_centers, *popt))
                plt.text(t, count.max() * 0.75, "Fidelity0: " + str(fidelity0) + "\nFidelity1: " + str(fidelity1) + 
                         "\nAverage: " + str((1 - filling_ratio) * fidelity0 + filling_ratio * fidelity1), va='top')
                plt.title("Emission values, fit, and threshold for trap (group) " + str(p_index))
                plt.vlines([t], 0, count.max(), colors='red')
                plt.legend(['Counts', 'Total fit', 'Detected threshold'])
                plt.savefig(os.path.join(histogram_path, "histogram_fit" + str(p_index) + ".png"))
                plt.clf()

        return first_peaks, second_peaks, fidelities, fidelities0, fidelities1, average_filling_ratio
    

    def calibrate_from_known(self, images, atom_locations : list[tuple[float,float]], psf = None,
                             average_closed_shutter_image = None, proj_shape : tuple[int,int] = None, 
                             min_cal_samples = None, psf_distance_mult = 2, camera_noise_reduction_method = "image"):
        """Function to calibrate the image analysis from known list of atom locations

        :param images: The images to be used for calibration. Should be iterable with each element being either a DataFrame or convertible to a numpy array
        :type images: list
        :param psf: The point-spread function if it is already known. Will be calibrated if None, defaults to None
        :type psf: numpy.array[float], optional
        :param average_closed_shutter_image: Average image without atoms or with closed shutter. Used to reduce camera noise if camera_noise_reduction_method = "image", defaults to None
        :type average_closed_shutter_image: numpy.array[float], optional
        :param proj_shape: Size of projection kernel. Will be set automatically based on atom spacing if not provided, defaults to None
        :type proj_shape: tuple[int,int], optional
        :param min_cal_samples: Number of samples to calibrate detection threshold. If min_cal_samples > len(images), \
            spatially close sites are grouped together for threshold calibration. Groups all sites together if None. Set to 0 if grouping is never desired, defaults to None
        :type min_cal_samples: int, optional
        :param psf_distance_mult: To consider a pixel for PSF calibration, the second-nearest atom must be at least psf_distance_mult times as distant as the nearest. \
            Bigger values reduce noise at PSf edge but reduce maximum meaningful PSF size, defaults to 2
        :type psf_distance_mult: float, optional
        :param camera_noise_reduction_method: Method of reducing camera noise, "border" to use median pixel value of images at the edges, "rowcol" to use median value of images per row and column, \
            "image" to use provided average_closed_shutter_image. If "image" and not average_closed_shutter_image provided, "rowcol" is used, defaults to "image"
        :type camera_noise_reduction_method: string, optional
        :raises AttributeError: Combination of attributes is not meaningful
        :return: List of detection threshold per site, [Empty-peak emission values, Occupied-peak emission values], Fidelity per atom site, 
            Fidelity0 (Fraction of empty sites detected as such) per atom site, Fidelity1 (Fraction of occupied sites detected as such) per atom site, Filling ratio
        :rtype: list[float], [list[float], list[float]], list[float], list[float], list[float], float
        """
        start_time = datetime.now()
        
        _, self.average_closed_shutter_image = self._get_average_images(images, camera_noise_reduction_method, average_closed_shutter_image)
        
        self.atom_locations = atom_locations

        if not hasattr(self, 'spacing') or self.spacing is None:
            min_dist = np.linalg.norm(np.array(self.atom_locations[0]) - np.array(self.atom_locations[1]))
            min_other_dist = 0
            ref_vector = np.array(self.atom_locations[0]) - np.array(self.atom_locations[1])
            for i in range(len(self.atom_locations)):
                for j in range(i + 1, len(self.atom_locations)):
                    vec = np.array(self.atom_locations[i]) - np.array(self.atom_locations[j])
                    dist = np.linalg.norm(vec)
                    if dist < min_dist:
                        ref_vector = vec / dist
                        min_dist = dist
                    if dist > min_other_dist:
                        min_other_dist = dist

            one_over_sqrt2 = 1 / np.sqrt(2)
            for i in range(len(self.atom_locations)):
                for j in range(i + 1, len(self.atom_locations)):
                    vec = np.array(self.atom_locations[i]) - np.array(self.atom_locations[j])
                    dist = np.linalg.norm(vec)
                    vec /= dist
                    if dist < min_other_dist and np.abs(np.dot(ref_vector, vec)) < one_over_sqrt2:
                        min_other_dist = dist
            self.spacing = (min_dist, min_other_dist)

        if psf is not None and isinstance(psf, np.array):
            self.psf = psf
        else:
            self._find_psf(images, self.average_closed_shutter_image, psf_distance_mult)
            if self.print_info:
                plt.imshow(self.psf)
                plt.title("Full scale PSF")
                plt.show()

        self._image_ref = atom_locations[0]

        self.solver = neutral_atom_image_analysis_cpp.ImageAnalysisProjection(self.psf, self.atom_locations)

        self._generate_and_set_projectors(proj_shape)
        
        parameters, atom_site_index_to_parameter_index = self._find_atom_site_groupings(images, min_cal_samples)

        # Reconstruct all test images to find best threshold
        start_time_reconstruct = datetime.now()
        for image in images:
            if isinstance(image, DataFrame):
                image_np = image.to_numpy(np.float64)
            else:
                image_np = np.array(image).astype(np.float64)
            result = self._reconstruct(image_np)
            for i in range(len(self.atom_locations)):
                parameters[atom_site_index_to_parameter_index[i]].append(result[i])
        if self.print_info:
            print("All images reconstructed within " + str((datetime.now() - start_time_reconstruct).total_seconds() * 1e3) + "ms")

        first_peak, second_peak, fidelities, fidelities0, fidelities1, filling_ratio = \
            self._calibrate_threshold(parameters, atom_site_index_to_parameter_index)
        if self.print_info:
            print("F0 avg: " + str(np.average(fidelities0)))
            print("F1 avg: " + str(np.average(fidelities1)))
            print("F avg: " + str(np.average(fidelities)))
            print("Calibration finished, total time: " + str((datetime.now() - start_time).total_seconds() * 1e3) + "ms")

        return self.threshold, [first_peak, second_peak], fidelities, fidelities0, fidelities1, filling_ratio
    

    def calibrate(self, images, average_closed_shutter_image = None, proj_shape : tuple[int,int] = None, 
        min_cal_samples = None, site_detection_threshold = 0.2, extend_locations_to_fov = False, 
        psf_distance_mult = 2, camera_noise_reduction_method = "image", angle_guesses : tuple[int,int] = (90, 0),
        optimize_locations_individually = False, remove_sites_under_fit_height_percentile = None, histogram_path = None):
        """Function to calibrate the image analysis

        :param images: The images to be used for calibration. Should be iterable with each element being either a DataFrame or convertible to a numpy array
        :type images: list
        :param average_closed_shutter_image: Average image without atoms or with closed shutter. Used to reduce camera noise if camera_noise_reduction_method = "image", defaults to None
        :type average_closed_shutter_image: numpy.array[float], optional
        :param proj_shape: Size of projection kernel. Will be set automatically based on atom spacing if not provided, defaults to None
        :type proj_shape: tuple[int,int], optional
        :param min_cal_samples: Number of samples to calibrate detection threshold. If min_cal_samples > len(images), \
            spatially close sites are grouped together for threshold calibration. Groups all sites together if None. Set to 0 if grouping is never desired, defaults to None
        :type min_cal_samples: int, optional
        :param site_detection_threshold: Fraction of maximum projection peak height to assume traps, defaults to 0.2
        :type site_detection_threshold: int, optional
        :param extend_locations_to_fov: If set to True, extend atom location grid in all directions within image coordinates, defaults to False
        :type extend_locations_to_fov: bool, optional
        :param psf_distance_mult: To consider a pixel for PSF calibration, the second-nearest atom must be at least psf_distance_mult times as distant as the nearest. \
            Bigger values reduce noise at PSf edge but reduce maximum meaningful PSF size, defaults to 2
        :type psf_distance_mult: float, optional
        :param camera_noise_reduction_method: Method of reducing camera noise, "border" to use median pixel value of images at the edges, "rowcol" to use median value of images per row and column, \
            "image" to use provided average_closed_shutter_image. If "image" and not average_closed_shutter_image provided, "rowcol" is used, defaults to "image"
        :type camera_noise_reduction_method: string, optional
        :param angle_guesses: Approximate angle of the two grid axes with respect to image axis, defaults to (90,0)
        :type angle_guesses: tuple[int,int], optional
        :param optimize_locations_individually: Whether to move each trap location individually to maximize alignment with local brightness peak. Otherwise only alignment with grid, default to False
        :type optimize_locations_individually: bool, optional
        :param remove_sites_under_fit_height_percentile: Remove trap locations from the grid where the detected brightness peak is very low. \
            All sites are removed where the peak height is below the percentile point function of remove_sites_under_fit_height_percentile and the fitted peak-height-parameters across all sites.\
            Does not remove any sites if None, defaults to None
        :type remove_sites_under_fit_height_percentile: float, optional
        :param histogram_path: Path used to save threshold histograms during calibration. Histograms not saved if None, defaults to None
        :type histogram_path: string, optional
        :raises AttributeError: Combination of attributes is not meaningful
        :return: List of detection threshold per site, [Empty-peak emission values, Occupied-peak emission values], Fidelity per atom site, 
            Fidelity0 (Fraction of empty sites detected as such) per atom site, Fidelity1 (Fraction of occupied sites detected as such) per atom site, Filling ratio
        :rtype: list[float], [list[float], list[float]], list[float], list[float], list[float], float
        """
        start_time = datetime.now()

        if images is None or not hasattr(images, '__iter__') or len(images) == 0:
            raise TypeError("List of images either not set, not iterable, or empty")
        
        average_filled_image, self.average_closed_shutter_image = self._get_average_images(images, camera_noise_reduction_method, average_closed_shutter_image)
        
        self._find_atom_locations(average_filled_image, site_detection_threshold, extend_locations_to_fov, 
                                  self.average_closed_shutter_image is not None, angle_guesses, optimize_locations_individually,\
                                  remove_sites_under_fit_height_percentile)
        if self.print_info:
            print("Atom_locations: " + str(self.atom_locations))
            plt.imshow(average_filled_image)
            plt.title("Average image with detected atom locations")
            if len(self.atom_locations) < 1000:
                for loc in self.atom_locations:
                    plt.plot(loc[1], loc[0], marker='x', color="red") 
            plt.show()
            print("Acquiring PSF")

        self._find_psf(images, self.average_closed_shutter_image, psf_distance_mult)
        if self.print_info:
            plt.imshow(self.psf)
            plt.title("Full scale PSF")
            plt.show()

        self.solver = neutral_atom_image_analysis_cpp.ImageAnalysisProjection(self.psf, self.atom_locations)

        self._generate_and_set_projectors(proj_shape)
        
        parameters, atom_site_index_to_parameter_index = self._find_atom_site_groupings(images, min_cal_samples)

        # Reconstruct all test images to find best threshold
        start_time_reconstruct = datetime.now()
        for image in images:
            if isinstance(image, DataFrame):
                image_np = image.to_numpy(np.float64)
            else:
                image_np = np.array(image).astype(np.float64)
            result = self._reconstruct(image_np)
            for i in range(len(self.atom_locations)):
                parameters[atom_site_index_to_parameter_index[i]].append(result[i])
        if self.print_info:
            print("All images reconstructed within " + str((datetime.now() - start_time_reconstruct).total_seconds() * 1e3) + "ms")

        first_peak, second_peak, fidelities, fidelities0, fidelities1, filling_ratio = \
            self._calibrate_threshold(parameters, atom_site_index_to_parameter_index, histogram_path)
        if self.print_info:
            print("F0 avg: " + str(np.average(fidelities0)))
            print("F1 avg: " + str(np.average(fidelities1)))
            print("F avg: " + str(np.average(fidelities)))
            print("Calibration finished, total time: " + str((datetime.now() - start_time).total_seconds() * 1e3) + "ms")

        return self.threshold, [first_peak, second_peak], fidelities, fidelities0, fidelities1, filling_ratio

    def _reconstruct(self, image):
        # Preprocess image
        if isinstance(image, DataFrame):
            image_np = image.to_numpy(np.float64)
        else:
            image_np = np.array(image).astype(np.float64)
        if np.isfortran(image_np):
            image_np = np.ascontiguousarray(image_np)
        image_np -= self.average_closed_shutter_image
        parameters = self.solver.reconstruct(image_np)
        return parameters

    def reconstruct(self, image):
        parameters = self._reconstruct(image)
        return parameters, [parameters[i] > self.threshold[i] for i in range(len(self.atom_locations))]
    