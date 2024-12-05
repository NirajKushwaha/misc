from .utils import *

def imshow_rgb(im, normalized_color_system=False):
    """
    Convert an image to an RGB array.

    Parameters
    ----------
    im : matplotlib.image.AxesImage
    normalized_color_system : bool, False
        If True, the RGB values will be in the range [0, 1]. Otherwise, they will be in the range [0, 255].

    Returns
    -------
    ndarray
    """

    data = im.get_array().data
    cmap = im.get_cmap()
    norm = im.norm

    rows, cols = data.shape
    rgb_values = np.zeros((rows, cols, 3))

    for i in range(rows):
        for j in range(cols):
            normalized_value = norm(data[i, j])  # Normalize the data value
            rgb_values[i, j] = cmap(normalized_value)[:3] # Map to RGB (ignore alpha)

    rgb_values_255 = (rgb_values * 255).astype(int)

    if(normalized_color_system):
        return rgb_values
    else:
        return rgb_values_255
