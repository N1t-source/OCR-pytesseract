import numpy as np
import cv2 as cv


img = cv.imread("images/test_image2.jpg")
small_img = cv.resize(img,None,fx=0.95,fy=0.95,interpolation=cv.INTER_AREA)
 

Img = small_img
GrayImg = cv.cvtColor(small_img, cv.COLOR_BGR2GRAY)
BlurredFrame = cv.GaussianBlur(GrayImg, (5, 5), 1)
CannyFrame = cv.Canny(BlurredFrame, 50, 150)

# Keep CannyFrame for viewing the image edges.
edge_contours, hierarchy = cv.findContours(
    CannyFrame,
    cv.RETR_LIST,
    cv.CHAIN_APPROX_SIMPLE
)

# Create a mask separating the bright ID card from the dark background.
# The card in this image is much brighter than the area around it.
_, CardMask = cv.threshold(GrayImg, 50, 255, cv.THRESH_BINARY)

# Find contours in the card mask rather than selecting small Canny edges
# from text, the flag, or the photograph.
card_contours, hierarchy = cv.findContours(
    CardMask,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

# Search for a large four-corner contour belonging to the card.
card_contour = None

for contour in sorted(card_contours, key=cv.contourArea, reverse=True):
    perimeter = cv.arcLength(contour, True)
    polygon = cv.approxPolyDP(contour, 0.02 * perimeter, True)

    # Ignore small objects. The card should cover most of the image.
    if len(polygon) == 4 and cv.contourArea(contour) > 0.5 * Img.shape[0] * Img.shape[1]:
        card_contour = polygon
        break

# Draw only the selected ID-card contour.
ContourFrame = Img.copy()

if card_contour is not None:
    cv.drawContours(ContourFrame, [card_contour], -1, (255, 0, 255), 4)
else:
    print("No four-corner card contour was found.")


cv.imshow('img', Img)
cv.imshow("GrayImg", GrayImg)
cv.imshow("BlurredFrame", BlurredFrame)
cv.imshow("CannyFrame", CannyFrame)
cv.imshow("CardMask", CardMask)
cv.imshow("ContourFrame", ContourFrame)


cv.waitKey(0)
