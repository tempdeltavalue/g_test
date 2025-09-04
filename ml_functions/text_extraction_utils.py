import pytesseract

pytesseract.pytesseract.tesseract_cmd = '/usr/local/opt/tesseract/bin/tesseract'


def pytesseract_get_text_from_image(image):
    text = pytesseract.image_to_string(image, lang='ukr')
    return text
