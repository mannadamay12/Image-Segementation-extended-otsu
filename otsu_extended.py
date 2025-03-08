import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class ExtendedOtsu:
    def __init__(self):
        pass
    
    def read_image(self, image_path):
        """
        Read a BMP image file
        """
        return np.array(Image.open(image_path))
    
    def rgb_to_grayscale(self, rgb_image):
        """
        Convert RGB image to grayscale using the formula:
        I = Round(0.299R + 0.587G + 0.114B)
        Note: We implement this manually as required
        """
        # Get image dimensions
        height, width, _ = rgb_image.shape
        
        # Create empty grayscale image
        grayscale = np.zeros((height, width), dtype=np.uint8)
        
        # Apply conversion formula
        for i in range(height):
            for j in range(width):
                r, g, b = rgb_image[i, j]
                gray_value = round(0.299 * r + 0.587 * g + 0.114 * b)
                grayscale[i, j] = gray_value
                
        return grayscale
    
    def compute_histogram(self, grayscale_image):
        """
        Compute the histogram of the grayscale image
        """
        # Initialize histogram with zeros
        histogram = np.zeros(256, dtype=np.int32)
        
        # Count pixels for each intensity
        height, width = grayscale_image.shape
        for i in range(height):
            for j in range(width):
                histogram[grayscale_image[i, j]] += 1
        
        # Normalize histogram
        normalized_histogram = histogram / (height * width)
        
        return normalized_histogram
    
    def compute_otsu_threshold(self, histogram):
        """
        Compute the optimal threshold using Otsu's method for 2 regions
        """
        # Number of possible gray levels
        n_levels = len(histogram)
        
        max_variance = 0
        optimal_threshold = 0
        optimal_variances = [0, 0]
        # For each possible threshold value
        for t in range(n_levels):
            # Background probability
            w_b = sum(histogram[:t+1])
            # Foreground probability
            w_f = sum(histogram[t+1:])
            
            # Skip if either region is empty
            if w_b == 0 or w_f == 0:
                continue
            
            # Compute background mean
            sum_b = 0
            for i in range(t+1):
                sum_b += i * histogram[i]
            mean_b = sum_b / w_b if w_b > 0 else 0
            
            # Compute foreground mean
            sum_f = 0
            for i in range(t+1, n_levels):
                sum_f += i * histogram[i]
            mean_f = sum_f / w_f if w_f > 0 else 0
            
            # Compute background variance
            var_b = 0
            for i in range(t+1):
                var_b += (i - mean_b) ** 2 * histogram[i]
            var_b = var_b / w_b if w_b > 0 else 0
            
            # Compute foreground variance
            var_f = 0
            for i in range(t+1, n_levels):
                var_f += (i - mean_f) ** 2 * histogram[i]
            var_f = var_f / w_f if w_f > 0 else 0
            
            # Compute total variance
            total_variance = w_b * var_b + w_f * var_f
            
            # Update optimal threshold if current variance is lower
            if t == 0 or total_variance < max_variance:
                max_variance = total_variance
                optimal_threshold = t
                optimal_variances = [var_b, var_f]
        
        return optimal_threshold, max_variance, optimal_variances
    
    def compute_multiple_thresholds(self, histogram, n_regions):
        """
        Compute multiple thresholds for segmentation into n_regions
        Uses an exhaustive search approach for demonstration purposes
        
        For real application, more efficient methods like dynamic programming
        should be used for n > 2
        """
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
                    variances, total_variance = self.compute_variance_for_thresholds(
                        histogram, [t1, t2])
                    if total_variance < min_variance:
                        min_variance = total_variance
                        optimal_thresholds = [t1, t2]
                        optimal_variances = variances
        
        elif n_regions == 4:
            # Exhaustive search for three thresholds
            for t1 in range(1, n_levels-2):
                for t2 in range(t1+1, n_levels-1):
                    for t3 in range(t2+1, n_levels):
                        variances, total_variance = self.compute_variance_for_thresholds(
                            histogram, [t1, t2, t3])
                        if total_variance < min_variance:
                            min_variance = total_variance
                            optimal_thresholds = [t1, t2, t3]
                            optimal_variances = variances
        
        return optimal_thresholds, min_variance, optimal_variances
    
    def compute_variance_for_thresholds(self, histogram, thresholds):
        """
        Compute the variances for multiple regions defined by thresholds
        """
        n_levels = len(histogram)
        n_regions = len(thresholds) + 1
        
        # Create region boundaries
        boundaries = [0] + thresholds + [n_levels]
        
        region_probabilities = []
        region_means = []
        region_variances = []
        
        # For each region
        for i in range(n_regions):
            start, end = boundaries[i], boundaries[i+1]
            
            # Compute region probability
            prob = sum(histogram[start:end])
            region_probabilities.append(prob)
            
            # Compute region mean
            sum_val = 0
            for j in range(start, end):
                sum_val += j * histogram[j]
            mean = sum_val / prob if prob > 0 else 0
            region_means.append(mean)
            
            # Compute region variance
            var = 0
            for j in range(start, end):
                var += (j - mean) ** 2 * histogram[j]
            var = var / prob if prob > 0 else 0
            region_variances.append(var)
        
        # Compute total variance
        total_variance = 0
        for i in range(n_regions):
            total_variance += region_probabilities[i] * region_variances[i]
        
        return region_variances, total_variance
    
    def segment_image(self, grayscale_image, thresholds):
        """
        Segment the image based on the thresholds
        """
        # Create a copy of the image for segmentation
        segmented = np.zeros_like(grayscale_image)
        
        # Add an upper bound for convenience
        effective_thresholds = thresholds + [256]
        n_regions = len(effective_thresholds)
        
        # Assign region values (using distinct gray levels)
        region_values = [int(255 * i / (n_regions)) for i in range(n_regions)]
        
        # Create the segmented image
        lower_bound = 0
        for i, t in enumerate(effective_thresholds):
            mask = (grayscale_image >= lower_bound) & (grayscale_image < t)
            segmented[mask] = region_values[i]
            lower_bound = t
        
        return segmented
    
    def process_image(self, image_path, n_regions=2):
        """
        Process an image using the Extended Otsu's method
        """
        # Read the image
        rgb_image = self.read_image(image_path)
        
        # Convert to grayscale
        grayscale = self.rgb_to_grayscale(rgb_image)
        
        # Compute histogram
        histogram = self.compute_histogram(grayscale)
        
        # Compute thresholds
        thresholds, total_variance, region_variances = self.compute_multiple_thresholds(
            histogram, n_regions)
        
        # Segment the image
        segmented = self.segment_image(grayscale, thresholds)
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_name = f"{base_name}-out-{n_regions}regions.bmp"
        
        # Save segmented image
        Image.fromarray(segmented).save(output_name)
        
        return grayscale, histogram, segmented, thresholds, total_variance, region_variances
    
    def visualize_results(self, grayscale, histogram, segmented, thresholds, 
                        total_variance, region_variances, title):
        """
        Visualize the results of the segmentation
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
    # Initialize the Extended Otsu processor
    processor = ExtendedOtsu()
    
    # List of test images (should be in the same directory)
    test_images = ["tiger1.bmp"]
    
    # Process each image with 2, 3, and 4 regions
    for image_path in test_images:
        for n_regions in [2, 3, 4]:
            grayscale, histogram, segmented, thresholds, total_variance, region_variances = \
                processor.process_image(image_path, n_regions)
            
            # Visualize results
            processor.visualize_results(
                grayscale, histogram, segmented, thresholds, 
                total_variance, region_variances, 
                f"{image_path} - {n_regions} Regions"
            )


if __name__ == "__main__":
    main()