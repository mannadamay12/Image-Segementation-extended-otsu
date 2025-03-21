import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import logging

# Set up logging to display info in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExtendedOtsu:
    def __init__(self):
        pass
    
    def read_image(self, image_path):
        """Read a BMP image file."""
        logging.info(f"Reading image from {image_path}")
        return np.array(Image.open(image_path))
    
    def rgb_to_grayscale(self, rgb_image):
        """
        Convert RGB image to grayscale using the formula:
        I = Round(0.299R + 0.587G + 0.114B)
        """
        logging.info("Converting image to grayscale")
        # Vectorized for efficiency
        grayscale = np.round(0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]).astype(np.uint8)
        return grayscale
    
    def compute_histogram(self, grayscale_image):
        """Compute the histogram of the grayscale image."""
        logging.info("Computing histogram")
        histogram, _ = np.histogram(grayscale_image, bins=256, range=(0, 256))
        normalized_histogram = histogram / grayscale_image.size
        return normalized_histogram
    
    def compute_otsu_threshold(self, histogram):
        """Compute the optimal threshold for 2 regions using Otsu's method."""
        logging.info("Computing Otsu threshold for 2 regions")
        n_levels = len(histogram)
        min_variance = float('inf')
        optimal_threshold = 0
        optimal_variances = [0, 0]
        
        for t in range(n_levels):
            w_b = sum(histogram[:t+1])  # Background weight
            w_f = sum(histogram[t+1:])  # Foreground weight
            if w_b == 0 or w_f == 0:
                continue
            sum_b = sum(i * histogram[i] for i in range(t+1))
            sum_f = sum(i * histogram[i] for i in range(t+1, n_levels))
            mean_b = sum_b / w_b
            mean_f = sum_f / w_f
            var_b = sum((i - mean_b) ** 2 * histogram[i] for i in range(t+1)) / w_b
            var_f = sum((i - mean_f) ** 2 * histogram[i] for i in range(t+1, n_levels)) / w_f
            total_variance = w_b * var_b + w_f * var_f
            if total_variance < min_variance:
                min_variance = total_variance
                optimal_threshold = t
                optimal_variances = [var_b, var_f]
        
        return optimal_threshold, min_variance, optimal_variances
    
    def compute_multiple_thresholds(self, histogram, n_regions):
        """
        Compute multiple thresholds for segmentation into n_regions.
        Uses exhaustive search for demonstration; dynamic programming could improve efficiency.
        """
        logging.info(f"Computing thresholds for {n_regions} regions")
        n_levels = len(histogram)
        min_variance = float('inf')
        optimal_thresholds = []
        optimal_variances = []
        
        if n_regions == 2:
            t, variance, variances = self.compute_otsu_threshold(histogram)
            return [t], variance, variances
        
        elif n_regions == 3:
            # Exhaustive search for two thresholds
            for t1 in range(1, n_levels-1):
                for t2 in range(t1+1, n_levels):
                    variances, total_variance = self.compute_variance_for_thresholds(histogram, [t1, t2])
                    if total_variance < min_variance:
                        min_variance = total_variance
                        optimal_thresholds = [t1, t2]
                        optimal_variances = variances
        
        elif n_regions == 4:
            # Exhaustive search for three thresholds
            for t1 in range(1, n_levels-2):
                for t2 in range(t1+1, n_levels-1):
                    for t3 in range(t2+1, n_levels):
                        variances, total_variance = self.compute_variance_for_thresholds(histogram, [t1, t2, t3])
                        if total_variance < min_variance:
                            min_variance = total_variance
                            optimal_thresholds = [t1, t2, t3]
                            optimal_variances = variances
        
        return optimal_thresholds, min_variance, optimal_variances
    
    def compute_variance_for_thresholds(self, histogram, thresholds):
        """Compute variances for multiple regions defined by thresholds."""
        n_levels = len(histogram)
        n_regions = len(thresholds) + 1
        boundaries = [0] + thresholds + [n_levels]
        region_probabilities = []
        region_means = []
        region_variances = []
        
        for i in range(n_regions):
            start, end = boundaries[i], boundaries[i+1]
            prob = sum(histogram[start:end])
            region_probabilities.append(prob)
            sum_val = sum(j * histogram[j] for j in range(start, end))
            mean = sum_val / prob if prob > 0 else 0
            region_means.append(mean)
            var = sum((j - mean) ** 2 * histogram[j] for j in range(start, end)) / prob if prob > 0 else 0
            region_variances.append(var)
        
        total_variance = sum(region_probabilities[i] * region_variances[i] for i in range(n_regions))
        return region_variances, total_variance
    
    def segment_image(self, grayscale_image, thresholds):
        """Segment the image based on the computed thresholds."""
        logging.info("Segmenting image")
        segmented = np.zeros_like(grayscale_image)
        effective_thresholds = thresholds + [256]
        n_regions = len(effective_thresholds)
        region_values = [int(255 * i / (n_regions)) for i in range(n_regions)]
        lower_bound = 0
        for i, t in enumerate(effective_thresholds):
            mask = (grayscale_image >= lower_bound) & (grayscale_image < t)
            segmented[mask] = region_values[i]
            lower_bound = t
        return segmented
    
    def process_image(self, image_path, n_regions):
        """Process an image using the Extended Otsu's method."""
        if n_regions not in [2, 3, 4]:
            logging.error(f"Invalid number of regions: {n_regions}. Must be 2, 3, or 4.")
            return None
        
        rgb_image = self.read_image(image_path)
        grayscale = self.rgb_to_grayscale(rgb_image)
        histogram = self.compute_histogram(grayscale)
        thresholds, total_variance, region_variances = self.compute_multiple_thresholds(histogram, n_regions)
        segmented = self.segment_image(grayscale, thresholds)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_name = f"{base_name}-out-{n_regions}regions.bmp"
        Image.fromarray(segmented).save(output_name)
        logging.info(f"Segmented image saved as {output_name}")
        
        return grayscale, histogram, segmented, thresholds, total_variance, region_variances
    
    def visualize_results(self, grayscale, histogram, segmented, thresholds, 
                          total_variance, region_variances, title):
        """Visualize the grayscale image, histogram, and segmented image."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(grayscale, cmap='gray')
        axes[0].set_title('Grayscale Image')
        axes[0].axis('off')
        axes[1].bar(range(256), histogram)
        axes[1].set_title('Histogram')
        axes[1].set_xlabel('Gray Level')
        axes[1].set_ylabel('Probability')
        for t in thresholds:
            axes[1].axvline(x=t, color='r', linestyle='--')
        axes[2].imshow(segmented, cmap='gray')
        axes[2].set_title('Segmented Image')
        axes[2].axis('off')
        variance_info = f"Total Variance: {total_variance:.6f}\n"
        for i, var in enumerate(region_variances):
            variance_info += f"Region {i+1} Variance: {var:.6f}\n"
        threshold_info = "Thresholds: " + ", ".join([str(t) for t in thresholds])
        plt.suptitle(f"{title}\n{threshold_info}\n{variance_info}")
        plt.tight_layout()
        plt.show()

def main():
    processor = ExtendedOtsu()
    test_images = ["tiger1.bmp", "data13.bmp", "basketballs.bmp"]
    
    # Interactive input for N
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