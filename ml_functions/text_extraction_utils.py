import pytesseract
from ml_functions.img_utils import find_receipt_contours, crop_receipts

def pytesseract_get_text_from_image(image, use_preprocessing=False):
    if use_preprocessing:
        found_contours, scale_factor = find_receipt_contours(image)
        cropped_receipts = crop_receipts(image, found_contours, scale_factor)
        text = ""
        for cr_receipt in cropped_receipts:
            text += pytesseract.image_to_string(cr_receipt, lang='ukr')
            text += "#########"

        return text 

    text = pytesseract.image_to_string(image, lang='ukr')
    return text
