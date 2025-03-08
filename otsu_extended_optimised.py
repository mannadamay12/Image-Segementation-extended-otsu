import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import logging  # Import the logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimizedExtendedOtsu:
    """
    An optimized implementation of the Extended Otsu's method using a pyramid approach
    for efficient multi-threshold segmentation.
    """
    
    def __init__(self):
        """Initialize the Extended Otsu processor."""
        pass
    
    def read_image(self, image_path):
        """Read a BMP image file."""
        logging.info(f"Reading image from {image_path}")
        return np.array(Image.open(image_path))
    
    def rgb_to_grayscale(self, rgb_image):
        """Convert RGB image to grayscale using vectorized operations."""
        logging.info("Converting image to grayscale")
        return np.round(0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]).astype(np.uint8)
    
    def compute_histogram(self, grayscale_image, bins=256):
        """Compute the histogram of the grayscale image."""
        logging.info("Computing histogram")
        # Initialize histogram with zeros
        histogram = np.zeros(bins, dtype=np.int32)
        
        # Count pixels for each intensity
        height, width = grayscale_image.shape
        for i in range(height):
            for j in range(width):
                histogram[grayscale_image[i, j]] += 1
        
        # Normalize histogram
        normalized_histogram = histogram / (height * width)
        
        return normalized_histogram
    
    def create_histogram_pyramid(self, hist):
        """
        Create a pyramid of histograms, each level compressed by a factor of 2
        from the previous level.
        """
        bins = len(hist)
        ratio = 2
        reductions = int(math.log(bins, ratio))
        
        hist_pyramid = []
        compression_factors = []
        
        for i in range(reductions):
            hist_pyramid.append(hist)
            # Compress histogram by combining adjacent bins
            reduced_hist = [sum(hist[j:j+ratio]) for j in range(0, bins, ratio)]
            hist = reduced_hist
            bins = bins // ratio
            compression_factors.append(2 if i > 0 else 1)
            
        return hist_pyramid, compression_factors
    
    def calculate_statistics(self, hist):
        """
        Calculate omega (cumulative probability) and mu (cumulative mean)
        for each possible threshold in the histogram.
        """
        bins = len(hist)
        N = float(sum(hist))
        
        # Calculate probability at each intensity level
        prob = [h / N for h in hist]
        
        # Calculate intensity-weighted probability
        weighted_prob = [i * prob[i] for i in range(bins)]
        
        # Calculate cumulative distributions
        omega = []
        mu = []
        
        omega_sum = 0
        mu_sum = 0
        
        for i in range(bins):
            omega_sum += prob[i]
            mu_sum += weighted_prob[i]
            
            omega.append(omega_sum)
            mu.append(mu_sum)
        
        # Total mean
        mu_total = mu_sum
        
        return omega, mu, mu_total
    
    def calculate_between_class_variance(self, omega, mu, mu_total, thresholds):
        """
        Calculate the between-class variance for a set of thresholds.
        Maximizing this variance is equivalent to minimizing within-class variance.
        """
        # Add boundary thresholds
        padded_thresholds = [-1] + thresholds + [len(omega) - 1]
        num_classes = len(padded_thresholds) - 1
        
        # Calculate total between-class variance
        total_variance = 0
        
        for i in range(num_classes):
            k1 = padded_thresholds[i]
            k2 = padded_thresholds[i + 1]
            
            # Calculate class probability and mean
            omega_k = omega[k2] - (omega[k1] if k1 >= 0 else 0)
            mu_k = mu[k2] - (mu[k1] if k1 >= 0 else 0)
            
            # Add to total variance
            if omega_k > 0:
                class_variance = omega_k * ((mu_k / omega_k - mu_total) ** 2)
                total_variance += class_variance
        
        return total_variance
    
    def search_around_thresholds(self, omega, mu, mu_total, thresholds, deviate=2):
        """
        Search around estimated thresholds within a small range to find
        better thresholds that maximize between-class variance.
        """
        bins = len(omega)
        best_variance = -1
        best_thresholds = thresholds.copy()
        
        # Generate candidates around current thresholds
        candidates = self._generate_threshold_candidates(thresholds, deviate, bins)
        
        for candidate in candidates:
            variance = self.calculate_between_class_variance(omega, mu, mu_total, candidate)
            
            if variance > best_variance:
                best_variance = variance
                best_thresholds = candidate.copy()
        
        return best_thresholds, best_variance
    def _generate_threshold_candidates(self, thresholds, deviate, bins):
        """
        Generate candidate threshold combinations within a deviate range
        of current thresholds, ensuring they remain in ascending order.
        """
        candidates = []
        def generate(current, remaining, prev_t):
            if not remaining:
                candidates.append(current[:])
                return
            t = remaining[0]
            for dt in range(-deviate, deviate + 1):
                new_t = t + dt
                if 0 <= new_t < bins and (prev_t is None or new_t > prev_t):
                    generate(current + [new_t], remaining[1:], new_t)
        generate([], thresholds, None)
        return candidates
    
    def compute_multiple_thresholds(self, hist, n_regions):
        """
        Compute optimal thresholds for segmenting an image into n_regions
        using the pyramid approach for efficiency.
        """
        logging.info(f"Computing Otsu threshold for {n_regions} regions")
        if n_regions <= 1:
            return [], 0, []
        
        # Number of thresholds is one less than number of regions
        k = n_regions - 1
        
        # Create histogram pyramid
        hist_pyramid, compression_factors = self.create_histogram_pyramid(hist)
        
        # Start with the smallest histogram
        smallest_hist = hist_pyramid[-1]
        omega, mu, mu_total = self.calculate_statistics(smallest_hist)
        
        # Initialize thresholds evenly across the smallest histogram
        smallest_bins = len(smallest_hist)
        initial_thresholds = [int((i + 1) * smallest_bins / (k + 1)) for i in range(k)]
        
        # Refine thresholds at the smallest level
        current_thresholds, _ = self.search_around_thresholds(
            omega, mu, mu_total, initial_thresholds, deviate=smallest_bins // 4
        )
        
        # Work back up the pyramid, refining thresholds at each level
        for i in range(len(hist_pyramid) - 2, -1, -1):
            current_hist = hist_pyramid[i]
            scaling = compression_factors[i]
            
            # Scale up thresholds for the next level
            scaled_thresholds = [t * scaling for t in current_thresholds]
            
            # Calculate statistics for current level
            omega, mu, mu_total = self.calculate_statistics(current_hist)
            
            # Refine thresholds at this level
            current_thresholds, _ = self.search_around_thresholds(
                omega, mu, mu_total, scaled_thresholds, deviate=scaling * 2
            )
        
        # Calculate final variances for the original histogram
        omega, mu, mu_total = self.calculate_statistics(hist)
        
        # Calculate within-group variances for each region
        region_variances = self._calculate_region_variances(hist, current_thresholds)
        
        # Calculate total variance (weighted sum of within-group variances)
        total_variance = sum(region_variances)
        
        return current_thresholds, total_variance, region_variances
    
    def _calculate_region_variances(self, hist, thresholds):
        """
        Calculate within-group variances for each region defined by thresholds.
        """
        bins = len(hist)
        N = float(sum(hist))
        
        # Add boundary thresholds
        padded_thresholds = [-1] + thresholds + [bins - 1]
        num_regions = len(padded_thresholds) - 1
        
        region_variances = []
        
        for i in range(num_regions):
            k1 = padded_thresholds[i] + 1  # Start of region (inclusive)
            k2 = padded_thresholds[i + 1]  # End of region (inclusive)
            
            # Calculate region statistics
            region_sum = sum(hist[k1:k2+1])
            region_prob = region_sum / N
            
            if region_prob > 0:
                # Calculate mean intensity in this region
                weighted_sum = sum(j * hist[j] for j in range(k1, k2+1))
                region_mean = weighted_sum / region_sum
                
                # Calculate variance in this region
                region_variance = sum(((j - region_mean) ** 2) * hist[j] for j in range(k1, k2+1))
                region_variance = region_variance / region_sum * region_prob
                
                region_variances.append(region_variance)
            else:
                region_variances.append(0)
        
        return region_variances
    
    def segment_image(self, grayscale_image, thresholds):
        """
        Segment the image based on the thresholds.
        """
        logging.info("Segmenting image")
        height, width = grayscale_image.shape
        segmented = np.zeros((height, width), dtype=np.uint8)
        
        # Number of regions
        n_regions = len(thresholds) + 1
        
        # Add upper bound for convenience
        padded_thresholds = [-1] + thresholds + [256]
        
        # Assign distinct gray values to each region
        region_values = [int(255 * i / (n_regions - 1)) for i in range(n_regions)]
        
        # Create the segmented image
        for i in range(n_regions):
            lower = padded_thresholds[i] + 1
            upper = padded_thresholds[i + 1] + 1
            
            # Create mask for pixels in this region
            if i == 0:
                mask = grayscale_image <= upper
            elif i == n_regions - 1:
                mask = grayscale_image > lower
            else:
                mask = (grayscale_image > lower) & (grayscale_image <= upper)
                
            # Apply the corresponding gray value
            segmented[mask] = region_values[i]
        
        return segmented
    
    def process_image(self, image_path, n_regions=2):
        """
        Process an image using the Extended Otsu's method with pyramid optimization.
        """
        # Read the image
        rgb_image = self.read_image(image_path)
        
        # Convert to grayscale
        grayscale = self.rgb_to_grayscale(rgb_image)
        
        # Compute histogram (as a list to match the pyramid implementation requirements)
        hist_array = self.compute_histogram(grayscale, bins=256)
        hist = [int(h * (grayscale.shape[0] * grayscale.shape[1])) for h in hist_array]
        
        # Compute thresholds
        thresholds, total_variance, region_variances = self.compute_multiple_thresholds(
            hist, n_regions
        )
        
        # Segment the image
        segmented = self.segment_image(grayscale, thresholds)
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_name = f"{base_name}-out-{n_regions}regions.bmp"
        
        # Save segmented image
        Image.fromarray(segmented).save(output_name)
        
        return grayscale, hist_array, segmented, thresholds, total_variance, region_variances
    
    def visualize_results(self, grayscale, histogram, segmented, thresholds, 
                          total_variance, region_variances, title):
        """
        Visualize the results of the segmentation.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original grayscale image
        axes[0].imshow(grayscale, cmap='gray')
        axes[0].set_title('Grayscale Image')
        axes[0].axis('off')
        
        # Plot histogram
        axes[1].bar(range(256), histogram)
        axes[1].set_title('Histogram')
        axes[1].set_xlabel('Gray Level')
        axes[1].set_ylabel('Probability')
        
        # Mark thresholds on histogram
        for t in thresholds:
            axes[1].axvline(x=t, color='r', linestyle='--')
        
        # Plot segmented image
        axes[2].imshow(segmented, cmap='gray')
        axes[2].set_title('Segmented Image')
        axes[2].axis('off')
        
        # Add overall title with variance information
        variance_info = f"Total Variance: {total_variance:.6f}\n"
        for i, var in enumerate(region_variances):
            variance_info += f"Region {i+1} Variance: {var:.6f}\n"
        
        threshold_info = "Thresholds: " + ", ".join([str(t) for t in thresholds])
        
        plt.suptitle(f"{title}\n{threshold_info}\n{variance_info}")
        plt.tight_layout()
        plt.show()


def main():
    # Initialize the Optimized Extended Otsu processor
    processor = OptimizedExtendedOtsu()
    
    # List of test images (should be in the same directory)
    test_images = ["tiger1.bmp"]
    # test_images = ["tiger1.bmp", "data13.bmp", "basketballs.bmp"]
    while True:
        n_regions_input = input("Enter the number of regions (2, 3, or 4) or type 'exit' to quit: ")
        if n_regions_input.lower() == 'exit':
            break
        try:
            n_regions = int(n_regions_input)
            if n_regions not in [2, 3, 4]:
                raise ValueError
        except ValueError:
            logging.error("Invalid input. Please enter 2, 3, or 4.")
            return
        
        # Process each image
        for image_path in test_images:
            result = processor.process_image(image_path, n_regions)
            if result:
                grayscale, histogram, segmented, thresholds, total_variance, region_variances = result
                processor.visualize_results(grayscale, histogram, segmented, thresholds, 
                                            total_variance, region_variances, 
                                            f"{image_path} - {n_regions} Regions")

if __name__ == "__main__":
    main()