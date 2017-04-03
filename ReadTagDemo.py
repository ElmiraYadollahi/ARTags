#Library import

import numpy as np
import cv2
from matplotlib import pyplot as plt

stepLevel=16
numDisp=16
bSize=11
stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=bSize)

# the encoding matrix
G = ['1101', '1011', '1000', '0111', '0100', '0010', '0001']
# the parity-check matrix
H = ['1010101', '0110011', '0001111']
Ht = ['100', '010', '110', '001', '101', '011', '111']
# the decoding matrix
R = ['0010000', '0000100', '0000010', '0000001']

def stereoMatch(textureImage):	
	cv2.imshow("map", textureImage)	
	grayTexture = cv2.cvtColor(textureImage, cv2.COLOR_BGR2GRAY)
	height, width = grayTexture.shape	
	#grayTexture = cv2.GaussianBlur(grayTexture,(5,5),0)
	grayTexture = cv2.medianBlur(grayTexture, 5) 
	imgL = grayTexture[0:height, 0:height/2-1]
	imgR = grayTexture[0:height, height/2:height-1]
	stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=bSize)
	disparity = stereo.compute(imgL,imgR)		
	return disparity

def find_squares(frame):	
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	img_filt=cv2.GaussianBlur(frame, (5,5), 0)
	img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	im2, contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
	return contours
	
def correctError(x):
	z = ''.join([str(bin(int(j, 2) & int(x, 2)).count('1') % 2) for j in H])
	if int(z, 2) > 0:
		e = int(Ht[int(z, 2) - 1], 2)
	else:
		e = 0	

	# correct the error
	if e > 0:
		x = list(x)
		x[e - 1] = str(1 - int(x[e - 1]))
		x = ''.join(x)

	p = ''.join([str(bin(int(k, 2) & int(x, 2)).count('1') % 2) for k in R])
	return p 

def readDisparity(disaprityMap):
	
	height, width = disaprityMap.shape		
	step = height/stepLevel	
	
	returnValue=''
	
	for h in xrange(0, height, step):	
		roi = disaprityMap[h:h+step, numDisp+bSize/2:width-1]			
		hist,bins = np.histogram(roi.ravel(),255,[1,200])
		for x in xrange(0, len(hist)):
			if hist[x] == hist.max():
				modeValue = bins[x]			
		if modeValue < 50:
			returnValue = returnValue + '0'
		else:
			returnValue = returnValue + '1'
	
	firstBit = returnValue[0]
	lastBit = returnValue[15]
	front = returnValue[1:8]
	back = returnValue[8:15]
	
	if firstBit is '0' and lastBit is '1':
		return '' + correctError(front) + correctError(back)
	else:	
		return 'NaN'

#Read video streams 
cap = cv2.VideoCapture(0) # put file name here to read from a video file
#cap.set(3,1920)
#cap.set(4,1080)
while(True):
	ret, image = cap.read()		
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	#cv2.imshow("gray", gray)

	edged = cv2.Canny(gray, 30, 200)

	# find contours in the edged image, keep only the largest
	# ones, and initialize our screen contour

	im2, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	# loop over our contours
	for c in cnts:
		if cv2.contourArea(c)>10000:  # remove small areas like noise etc
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.04 * peri, True)			
		    
			# if our approximated contour has four points, then
			# we can assume that we have found our screen
			if len(approx) == 4:
				screenCnt = approx
				#break


	if screenCnt is not None:
		
		pts = screenCnt.reshape(4, 2)
		rect = np.zeros((4, 2), dtype = "float32")
		
		# the top-left point has the smallest sum whereas the
		# bottom-right has the largest sum
		s = pts.sum(axis = 1)
		rect[0] = pts[np.argmin(s)]
		rect[2] = pts[np.argmax(s)]
		 
		# compute the difference between the points -- the top-right
		# will have the minumum difference and the bottom-left will
		# have the maximum difference
		diff = np.diff(pts, axis = 1)
		rect[1] = pts[np.argmin(diff)]
		rect[3] = pts[np.argmax(diff)]
		
		# now that we have our rectangle of points, let's compute
		# the width of our new image
		(tl, tr, br, bl) = rect
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		 
		# ...and now for the height of our new image
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		 
		# take the maximum of the width and height values to reach
		# our final dimensions
		maxWidth = max(int(widthA), int(widthB))
		maxHeight = max(int(heightA), int(heightB))
		 
		# construct our destination points which will be used to
		# map the screen to a top-down, "birds eye" view
		dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]], dtype = "float32")
		 
		# calculate the perspective transform matrix and warp
		# the perspective to grab the screen
		M = cv2.getPerspectiveTransform(rect, dst)
		warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
		
		
		dst = cv2.resize(warp, (544, 544))
		roi = dst[16:512+16, 16:512+16]	
		cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
		
		disparity = stereoMatch(roi)	
		#imC = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
		cv2.imshow("displayImage", 50*disparity)	
		
		textFound = readDisparity(disparity)
		
		# Write some Text
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image,textFound,((pts[0][0]+pts[1][0]+pts[2][0]+pts[3][0])/4 - 80, (pts[0][1]+pts[1][1]+pts[2][1]+pts[3][1])/4 +10), font, 1,(0,0,255),2)
	
	#image = cv2.resize(image, (800, 600))
	cv2.imshow("image", image)
	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap.release()




