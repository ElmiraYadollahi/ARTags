# Create QR code
#Library import
import numpy as np
import cv2
import qrcode
from matplotlib import pyplot as plt
from random import randint

OFFSET = 5
SIZE_SQUARE = 20
numDisp=16
bSize=7
stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=bSize)

encodedURL = "www.aut.ac.nz/minh"
textureURL = "mickey.png"
backgroundURL = "background.jpg"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=1,
    border=0,
)

def stereoMatch(textureImage, scale=True):		
	if scale:
		textureImage = cv2.resize(textureImage, (1024, 512))	
	height, width, shape = textureImage.shape	
	imgL = textureImage[0:height, 0:width/2]
	imgR = textureImage[0:height, width/2:width]	
	disparity = stereoMatch2pairs(imgL, imgR)		
	return disparity

def stereoMatch2pairs(imgL, imgR):	
	imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
	imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)	
	height, width = imgL.shape	
	
	imgLWithGap = np.zeros((height, width + bSize + numDisp), np.uint8)
	imgRWithGap = np.zeros((height, width + bSize + numDisp), np.uint8)
	
	for h in xrange( 0, height):
		imgLWithGap[h:] = imgL[h,0]
		imgRWithGap[h:] = imgR[h,0]		
	
	imgLWithGap[0:height,bSize + numDisp:bSize + numDisp + width] = imgL
	imgRWithGap[0:height,bSize + numDisp:bSize + numDisp + width] = imgR	
	
	disparity = stereo.compute(imgLWithGap,imgRWithGap)		
	
	disparityNoGap = np.zeros((height, width), np.uint8)
	disparityNoGap = disparity[0:height,bSize + numDisp:bSize + numDisp+width]	
	
	return disparityNoGap

def decorateTexture(imgTexture):
	height, width, depth = imgTexture.shape 
	disparity = stereoMatch2pairs(imgTexture, imgTexture)
	height, width = disparity.shape
	blankIdentificationImg = np.zeros((height,width,1), np.uint8) #with 1 chanels
	for h in xrange( 0, height):
		for w in xrange(0, width):
			if disparity[h][w] < 0:
				blankIdentificationImg[h][w] = 255 #white							
	
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(blankIdentificationImg,cv2.MORPH_OPEN,kernel, iterations = 2)

	sure_bg = cv2.dilate(opening,kernel,iterations=5)
	unknown = sure_bg.copy()
	sure_bg = cv2.blur(sure_bg,(35,35))	
	colourBGOrig = cv2.imread(backgroundURL, cv2.IMREAD_COLOR);
	heightBG, widthBG, depthBG = colourBGOrig.shape
	
	#build tiled colour background
	colourBG = np.zeros((height, width, 3), np.uint8)
	for h in xrange (0, height):
		for w  in xrange (0, width):
			colourBG[h,w] = colourBGOrig[h%heightBG, w%widthBG]	
	
	for h in xrange( 0, height):
		for w in xrange(0, width):
			if sure_bg[h][w] > 0 :
				value = 1.0 * sure_bg[h][w] / 255.0
				imgTexture[h][w] = value * colourBG[h][w] + (1-value)*imgTexture[h][w]		
	return imgTexture

qr.add_data(encodedURL)
qr.make(fit=True)
img = qr.make_image()
qrArray = np.array(img)
qrWidth, qrHeight = qrArray.shape

QR_NUMBER = qrWidth

imgTexture = cv2.imread(textureURL, cv2.IMREAD_COLOR)
imgTexture = decorateTexture(imgTexture)

height, width, depth = imgTexture.shape 

ConventionalSide = (QR_NUMBER+2)*SIZE_SQUARE
hRatio = 1.0*height/ConventionalSide
wRatio = 1.0*width/ConventionalSide

imgTexture = cv2.resize(imgTexture, (ConventionalSide, ConventionalSide))
height, width, depth = imgTexture.shape 

sbsImage = np.zeros((height, 2*width, 3), np.uint8) #with 3 chanels # img2 = img1.copy()
sbsHeight, sbsWidth, sbsDepth = sbsImage.shape 

sbsImage[0:height, width/2:3*width/2] = imgTexture
sbsImage[0:height, 0:width/2] = imgTexture[0:height, width/2:width]
sbsImage[0:height, 3*width/2:2*width] = imgTexture[0:height, 0:width/2]

sbsImageNew = sbsImage.copy()
for h in xrange(0, QR_NUMBER):	
	for w in xrange(0, QR_NUMBER/2):			
		y0 = SIZE_SQUARE + SIZE_SQUARE*h
		y1 = SIZE_SQUARE + SIZE_SQUARE*h + SIZE_SQUARE
		x0 = SIZE_SQUARE + SIZE_SQUARE*w + width
		x1 = SIZE_SQUARE + SIZE_SQUARE*w + SIZE_SQUARE + width
		patternBox = sbsImage[y0:y1, x0:x1]					
		
		sbsImageNew[y0:y1, x0-width:x1-width] = patternBox	
		if qrArray[h][w] == False:	
			sbsImageNew[y0:y1, x0-width+OFFSET:x1-width+OFFSET] = patternBox			
			
	for w in xrange(QR_NUMBER/2, QR_NUMBER):			
		y0 = SIZE_SQUARE + SIZE_SQUARE*h
		y1 = SIZE_SQUARE + SIZE_SQUARE*h + SIZE_SQUARE
		x0 = width + SIZE_SQUARE + SIZE_SQUARE*w
		x1 = width + SIZE_SQUARE + SIZE_SQUARE*w + SIZE_SQUARE
		patternBox = sbsImage[y0:y1, x0:x1]
					
			
		sbsImageNew[y0:y1, x0-width:x1-width] = patternBox	
		if qrArray[h][w] == False:	
			sbsImageNew[y0:y1, x0-OFFSET:x1-OFFSET] = patternBox
		else:
			sbsImageNew[y0:y1, x0:x1] = patternBox	
			

heightSBS, widthSBS, depthSBS = sbsImageNew.shape 

sbsQR = cv2.resize(sbsImageNew, (int(wRatio*widthSBS), int(hRatio*heightSBS)))
depth = stereoMatch(sbsQR)
ret,thresh1 = cv2.threshold(depth,0,255,cv2.THRESH_BINARY)

kernel = np.ones((7,7),np.uint8)
closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

cv2.imshow('sbsQR',sbsQR)
plt.imshow(255-closing, cmap = 'gray')
plt.show()


