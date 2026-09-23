import numpy as np
import cv2 as cv


img = cv.imread("images/test_image2.jpg")
small_img = cv.resize(img,None,fx=0.75,fy=0.75,interpolation=cv.INTER_AREA)
 

Img = small_img
GrayImg = cv.cvtColor(small_img, cv.COLOR_BGR2GRAY)
BlurredFrame = cv.GaussianBlur(GrayImg, (3, 3), 1)


cv.imshow('img', Img)
cv.imshow("GrayImg", GrayImg)
cv.imshow("BlurredFrame", BlurredFrame)


cv.waitKey(0)