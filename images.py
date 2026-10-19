from utils import *
import pymupdf

def pdf_to_img(FILEPATH, page_number=0, dpi=150):
    """
    Loads a PDF file and converts the specified page to an image.

    Parameters
    ----------
    FILEPATH : str
        The path to the PDF file.
    page_number : int, 0
        The page number to convert.
    dpi : int, 150
        The resolution of the output image.

    Returns
    -------
    PIL.Image.Image
        The image of the specified page.
    """

    doc = pymupdf.open(FILEPATH)

    pix = doc[page_number].get_pixmap(dpi=dpi)
    image = pix.pil_image()

    return image