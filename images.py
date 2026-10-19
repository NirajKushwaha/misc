from utils import *
import pymupdf

def pdf_to_img(FILEPATH, page_number=0):
    """
    Loads a PDF file and converts the specified page to an image.

    Parameters
    ----------
    FILEPATH : str
        The path to the PDF file.
    page_number : int, 0
        The page number to convert.

    Returns
    -------
    PIL.Image.Image
        The image of the specified page.
    """

    doc = pymupdf.open(FILEPATH)

    pix = doc[page_number].get_pixmap(dpi=150)
    image = pix.pil_image()

    return image