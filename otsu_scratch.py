import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # import scipy.misc as sm
    # import image.io
    from matplotlib import pyplot
    return (pyplot,)


@app.cell
def _(argmin, no_of_bins):
    def otsu_orig(hist, total):
        no_of_bins - len(hist) #=256
        intra_class_variances = []
        for threshold in range(0, no_of_bins): # calc weight and var on bg
            sum_bg = float(sum(hist[0:threshold]))
            weight_bg = sum_bg/total
            mean_bg = 0.0
            var_bg = 0.0
            if sum_bg > 0.0: # avoid / 0
                for x in range(0, threshold):
                    mean_bg += x*hist[x]
                mean_bg/=sum_bg
                for x in range(0, threshold):
                    var_bg += [x - mean_bg] ** 2 * hist[x]
                var_bg/=sum_bg

            sum_fg = float(sum(hist[threshold:no_of_bins])) # repeat the same procedure for fg
            weight_fg = sum_fg/total
            mean_fg = 0.0
            var_fg = 0.0

            if sum_fg > 0.0: # avoid / 0
                for x in range(threshold, no_of_bins):
                    mean_fg += x*hist[x]
                mean_fg/=sum_fg
                for x in range(threshold, no_of_bins):
                    var_fg += [x - mean_fg] ** 2 * hist[x]
                var_fg/=sum_fg

            intra_class_variances.append(weight_bg*var_bg + weight_fg*var_fg) # calc var within fg and bg

        return argmin(intra_class_variances) - 1 # output threshold with min intra class variance btw fg and bg
    return (otsu_orig,)


@app.cell
def _(np, otsu_orig, pyplot, shape, sm):
    def main():
        img = sm.imread('/Users/ad12/Documents/NYU/CV/project-1-otsu/test_input/Bright_green_tree_-_Waikato.jpg')
        img = sm.imresize(img, (1944/4, 2592/4))

        grayscale = img.np.dot([0.299, 0.587, 0.114])
        rows, cols = shape(grayscale)

        hist = np.histogram(grayscale, 256)[0]

        thresh = otsu_orig(hist, rows*cols)

        figure = pyplot.figure(figsize = (14,6))
        figure.canvas.set_window_title('Otsu from scratch')

        axes = figure.add_subplot(121)
        axes.set_title('Original image')
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.imshow(img, cmap='Greys_r')

        axes = figure.add_subplot(122)
        axes.set_title('Otsu thresholding on image')
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.imshow(grayscale >= thresh, cmap='Greys_r')

        pyplot.show()
    return (main,)


@app.cell
def _(main):
    if __name__ == '__main__':
        main()
    return