import easyocr

reader = easyocr.Reader(['en'])


def read_plate(image_path):

    result = reader.readtext(image_path)

    text = ""

    for detection in result:
        text += detection[1]

    return text