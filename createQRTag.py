# Create QR code
#Library import
import numpy as np
import cv2
import qrcode
from matplotlib import pyplot as plt
from random import randint

OFFSET = 5
SIZE_SQUARE = 20
STD_THRESHOLD = 30
numDisp=16
bSize=7

encodedURL = "www.aut.ac.nz/minh"
textureURL = "mickey.png"
backgroundURL = "mickyMouseBG.jpg"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=1,
    border=0,
)

def stereoMatch(textureImage, scale=True):		
	if scale:
		textureImage = cv2.resize(textureImage, (1024, 512))
	grayTexture = cv2.cvtColor(textureImage, cv2.COLOR_BGR2GRAY)
	height, width = grayTexture.shape	
	imgL = grayTexture[0:height, 0:width/2]
	imgR = grayTexture[0:height, width/2:width]
	stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=bSize)
	disparity = stereo.compute(imgL,imgR)		
	return disparity

def stereoMatch2pairs(imgL, imgR):	
	imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
	imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
	stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=bSize)
	disparity = stereo.compute(imgL,imgR)		
	return disparity

def decorateTexture(imgTexture):
	height, width, depth = imgTexture.shape 
	disparity = stereoMatch2pairs(imgTexture, imgTexture)
	height, width = disparity.shape
	blankIdentificationImg = np.zeros((height,width,1), np.uint8) #with 1 chanels
	for h in xrange( bSize/2, height-bSize/2):
		for w in xrange(numDisp+bSize/2, width-bSize/2):
			if disparity[h][w] < 0:
				blankIdentificationImg[h][w] = 255 #white							
	
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(blankIdentificationImg,cv2.MORPH_OPEN,kernel, iterations = 2)

	sure_bg = cv2.dilate(opening,kernel,iterations=5)
	unknown = sure_bg.copy()
	sure_bg = cv2.blur(sure_bg,(35,35))	
	colourBG = cv2.imread(backgroundURL);
	colourBG = cv2.resize(colourBG, (width, height))

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

cv2.imshow('sbsImage',sbsImage)

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
		#else:
		#	sbsImageNew[y0:y1, x0:x1] = patternBox		
			
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
			
cv2.imshow('sbsImage',sbsImageNew)

heightSBS, widthSBS, depthSBS = sbsImageNew.shape 

print "ratio ", wRatio
sbsQR = cv2.resize(sbsImageNew, (int(wRatio*widthSBS), int(hRatio*heightSBS)))
#depth = stereoMatch(sbsImageNew)
depth = stereoMatch(sbsQR)

ret,thresh1 = cv2.threshold(depth,0,255,cv2.THRESH_BINARY)

kernel = np.ones((7,7),np.uint8)
#opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

cv2.imshow('sbsQR',sbsQR)
#cv2.imshow('thresh1',thresh1)
#for x in thresh1:
#	print x
plt.imshow(255-closing, cmap = 'gray')
plt.show()


