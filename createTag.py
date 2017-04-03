import numpy as np
import cv2
from random import randint
from matplotlib import pyplot as plt
import random

# Hamming(7,4) Error Correction Code
# https://en.wikipedia.org/wiki/Hamming(7%2C4)

# the encoding matrix
G = ['1101', '1011', '1000', '0111', '0100', '0010', '0001']
# the parity-check matrix
H = ['1010101', '0110011', '0001111']
Ht = ['100', '010', '110', '001', '101', '011', '111']
# the decoding matrix
R = ['0010000', '0000100', '0000010', '0000001']

mainImage = "micky.png"
backgroundImage = "mickyMouseBG.jpg"
stringCode = '10100110' #En code One byte using Hamming code --> 16 bytes, with top is zero and bottom is one.

stepLevel=16
numDisp=16
bSize=11



#divide the string by two
str1 = stringCode[:4]
str2 = stringCode[4:8]

#encode string 1
xStr1 = ''.join([str(bin(int(i, 2) & int(str1, 2)).count('1') % 2) for i in G])
#encode string 2
xStr2 = ''.join([str(bin(int(i, 2) & int(str2, 2)).count('1') % 2) for i in G])

finalEncodeString = '0'+xStr1+xStr2+'1'
print finalEncodeString

def createStereogram(textureImage):	
	height, width, depth = textureImage.shape	
	smallHeight = height/stepLevel	
	for h in xrange(0, height):
		k = 0
		if finalEncodeString[h / smallHeight] == '1':
			k = 12		
			
		for w in xrange(0, height/4 + k):			
			if w < height/4:
				textureImage[h, w] = textureImage[h,height/2 + w - k]		
			textureImage[h, 3*height/4 + w - k] = textureImage[h,height/4 + w]	
	return textureImage
	
def stereoMatch(textureImage):		
	grayTexture = cv2.cvtColor(textureImage, cv2.COLOR_BGR2GRAY)
	height, width = grayTexture.shape	
	imgL = grayTexture[0:height, 0:height/2-1]
	imgR = grayTexture[0:height, height/2:height-1]
	stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=bSize)
	disparity = stereo.compute(imgL,imgR)		
	return disparity


#begin main code
tempImage = cv2.imread(mainImage);
height, width, depth = tempImage.shape

#assume that height is longer than width

textureImage = np.zeros((height,height,3), np.uint8)

#if height longer than width
wGap = (height - width)/2;

if wGap < 0:
	wGap = 0
	tempImage = cv2.resize(tempImage, (height, height))
	height, width, depth = tempImage.shape

for h in xrange(0, height):
	for w in xrange(0, width):
		textureImage[h, wGap + w] = tempImage[h,w]		
		
cv2.imshow('textureImage Original 1',textureImage)
		
for h in xrange(0, height):
	for w in xrange(0, height/4):
		textureImage[h, w] = textureImage[h,height/2+w]
		textureImage[h, 3*height/4 + w] = textureImage[h,height/4+w]

textureImage = cv2.resize(textureImage, (512, 512))
height, width, depth = textureImage.shape
originalWidth = width

textureImage = createStereogram(textureImage)			

originalImage = textureImage.copy()

cv2.imshow('textureImage Original 2',textureImage)

textureImage = cv2.GaussianBlur(textureImage,(5,5),0)

cv2.imshow('textureImage',textureImage)
cv2.imwrite('tag.png', textureImage)

disparity = stereoMatch(textureImage)

height, width = disparity.shape

blankIdentificationImg = np.zeros((height,width,1), np.uint8) #with 1 chanels

for h in xrange( bSize/2, height-bSize/2):
	for w in xrange(numDisp+bSize/2, width-bSize/2):
		if disparity[h][w] < 0:
			blankIdentificationImg[h][w] = 255 #white

cv2.imshow('blankIdentificationImg 1',blankIdentificationImg)
cv2.imwrite('blankIdentificationImg.png', blankIdentificationImg)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(blankIdentificationImg,cv2.MORPH_OPEN,kernel, iterations = 2)
    
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=5)

unknown = sure_bg.copy()

sure_bg = cv2.blur(sure_bg,(35,35))

cv2.imshow('sure_bg',sure_bg)
cv2.imwrite('sure_bg.png', sure_bg)
cv2.imshow('unknown',unknown)
cv2.imwrite('unknown.png', unknown)

colourBG = cv2.imread(backgroundImage);
colourBG = cv2.resize(colourBG, (512, 512))

cv2.imshow('colourBG',colourBG)

# draw unknow region on original image
height, width = disparity.shape
for h in xrange( 0, height):
	for w in xrange(0, width):
		if sure_bg[h][w] > 0 :
			value = 1.0 * sure_bg[h][w] / 255.0
			originalImage[h][w] = value * colourBG[h][w] + (1-value)*originalImage[h][w]
			originalImage[h][w+originalWidth/2] = value * colourBG[h][w+originalWidth/2] + (1-value)*originalImage[h][w+originalWidth/2]

cv2.imshow('originalImage',originalImage)
cv2.imwrite('originalRedImage.png', originalImage)

textureImage = createStereogram(originalImage)
cv2.imshow('finalTextureImage', textureImage)
cv2.imwrite('finalTextureImage.png', textureImage)

newDisparity = stereoMatch(textureImage)

#build border
border = 16
height, width, depth = textureImage.shape #Total pixel number: img.size
borderImage = img = np.zeros((height+2*border,width+2*border,3), np.uint8) 
borderImage[border:height+border,border:width+border]=textureImage
cv2.imshow('borderImage',borderImage)
cv2.waitKey(1000)

