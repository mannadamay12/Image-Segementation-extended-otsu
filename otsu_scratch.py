import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
import imageio.v2 as imageio

def otsu_orig(hist, total):
    no_of_bins = len(hist)  # Should be 256 for 8-bit grayscale
    intra_class_variances = []
    
    for threshold in range(0, no_of_bins):
        # Background calculations
        sum_bg = float(sum(hist[0:threshold]))
        weight_bg = sum_bg / total
        mean_bg = 0.0
        var_bg = 0.0
        if sum_bg > 0.0:
            for x in range(0, threshold):
                mean_bg += x * hist[x]
            mean_bg /= sum_bg
            for x in range(0, threshold):
                var_bg += (x - mean_bg) ** 2 * hist[x]
            var_bg /= sum_bg

        # Foreground calculations
        sum_fg = float(sum(hist[threshold:no_of_bins]))
        weight_fg = sum_fg / total
        mean_fg = 0.0
        var_fg = 0.0
        if sum_fg > 0.0:
            for x in range(threshold, no_of_bins):
                mean_fg += x * hist[x]
            mean_fg /= sum_fg
            for x in range(threshold, no_of_bins):
                var_fg += (x - mean_fg) ** 2 * hist[x]
            var_fg /= sum_fg

        # Intra-class variance
        intra_class_variances.append(weight_bg * var_bg + weight_fg * var_fg)
    
    # Return threshold minimizing intra-class variance
    return np.argmin(intra_class_variances)

def main():
    # Load and resize image
    img = imageio.imread('tiger1.bmp')
    img = resize(img, (img.shape[0] // 4, img.shape[1] // 4), anti_aliasing=True)
    
    # Convert to grayscale
    grayscale = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    grayscale = np.round(grayscale).astype(np.uint8)
    
    # Get dimensions and histogram
    rows, cols = grayscale.shape
    hist = np.histogram(grayscale, 256, range=(0, 256))[0]
    
    # Compute threshold
    thresh = otsu_orig(hist, rows * cols)
    
    # Plotting
    figure = plt.figure(figsize=(14, 6))
    figure.suptitle('Otsu from Scratch')
    
    # Original image
    axes = figure.add_subplot(121)
    axes.set_title('Original Image')
    axes.imshow(img)
    axes.axis('off')
    
    # Thresholded image
    axes = figure.add_subplot(122)
    axes.set_title('Otsu Thresholding on Image')
    axes.imshow(grayscale >= thresh, cmap='gray')
    axes.axis('off')
    
    plt.show()


if __name__ == '__main__':
    main()