import PIL.Image
import pytesseract

test_image = PIL.Image.open("images/test_image2.jpg")

test_image_text= pytesseract.image_to_string(test_image)
print(test_image_text)