import numpy as np
import cv2


def rgbToGrey(arr):
    greyVal = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140])
    return np.round(greyVal).astype(np.uint8)


def getEdge(greyImg):
    edgeH = cv2.Sobel(greyImg,cv2.CV_64F,1,0,ksize=3)
    edgeV = cv2.Sobel(greyImg,cv2.CV_64F,0,1,ksize=3)
    
    return np.sqrt(np.square(edgeH) + np.square(edgeV))


def findCostArr(edgeImg):
    row,col = edgeImg.shape
    cost = np.zeros(edgeImg.shape)
    cost[row-1,:] = edgeImg[row-1,:]
    
    for i in range(row-2,-1,-1):
        for j in range(col):
            left,right = max(j-1,0),min(col,j+1)
            cost[i][j] = edgeImg[i][j] + cost[i+1,left:right].min()
                
    return cost


def findSeam(cost):
    
    row,col = cost.shape
    
    path = []
    j = cost[0].argmin()
    path.append(j)
    
    for i in range(row-1):
        left,right = max(j-1,0),min(col,j+2)
        j = max(j-1,0)+cost[i+1,left:right].argmin()
        path.append(j)

    return path


def removeSeam(img,path):
    row,col,_ = img.shape
    newImg = np.zeros((row,col,3))
    for i,j in enumerate(path):
        newImg[i,0:j,:] = img[i,0:j,:]
        newImg[i,j:col-1,:] = img[i,j+1:col,:]
    return newImg[:,:-1,:].astype(np.uint8)


def drawSeam(img,path):
    row,col,_ = img.shape
    for i,j in enumerate(path):
        img[i,j,:] = [0, 255, 0]
        