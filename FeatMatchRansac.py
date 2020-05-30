import cv2
import numpy as np
from matplotlib import pyplot as plt
simA1 =cv2.imread('transA.jpg')
simA = cv2.imread('transA.jpg')
simB1 = cv2.imread('transB.jpg')
simB = cv2.imread('transB.jpg')


###returns an image with corners marked in red (image) and local maxima values of corner locations
def H(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,7,0.04)
    dst = cv2.dilate(dst,None)
    image[dst>(0.1*dst.max())]=[0,0,255]
    cv2.imwrite('dst.png', dst) 
    cv2.imshow('dst',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image, dst

### a more accurate version of above
def Hacc(image):
    cv2.imshow('Hacc',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    image[res[:,1],res[:,0]]=[0,0,255]
    image[res[:,3],res[:,2]] = [0,255,0]
    cv2.imwrite('subpix.png',image)
    subpix = cv2.imread('subpix.png')
    cv2.imshow('subpix.png',subpix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return subpix


def drawlines(img1, img2, ptsa, ptsb):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1.0
    FONT_THICKNESS = 2
    bg_color = (255, 255, 255)
    label_color = (0, 0, 0)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #r, c = img1g.shape[0], img1g.shape[1]
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    

    cc = -1
    for pt1 in ptsa:
        vis = np.concatenate((img1, img2), axis=1)
        vis2 = vis
        cv2.imshow('vis', vis2)
        cv2.waitKey(300)
        cv2.destroyAllWindows()
        cc+=1
        bc = -1
        for pt2 in ptsb:
            
            bc+=1
            #print('location of pt1, pt2', pt1, pt2)
            pixA = img1[int(pt1[0]), int(pt1[1])]
            pixB= img2[int(ptsb[bc][0]), int(ptsb[bc][1])]
            #print('pixel color', pixA[0], pixB[0])
            x0, y0 = pt1[1], pt1[0]
            x1, y1 = pt2[1], pt2[0]
            text1 = ('pt1 X is %d') % int(x0)
            text3 = ('pt1 y is %d') % y0
            
            text2 = ('pt2 x is %d') % x1
            text4 = ('pt2 y is %d') % y1
            vis2 = cv2.circle(vis2, (int(pt2[1]+640), int(pt2[0])), 1, (255,0,0), 2)
            
            vis2 = cv2.circle(vis2, (int(pt1[1]), int(pt1[0])), 1, (0,255,0), 2)
            
            #print('pt1x, y', x1, y1)
            if pixA[0] == pixB[0]:# and pix1G == pix2G and pix1R == pix2R:
                cv2.putText(vis2, text1, (50,80), 2, 2,color=(2,250,0), thickness=2)
                cv2.putText(vis2, text3, (50,150), 2, 2,color=(2,250,0), thickness=2)
                cv2.putText(vis2, text2, (50,200), 2, 2,color=(0,0,255), thickness=2)
                cv2.putText(vis2, text4, (50,270), 2, 2,color=(0,0,255), thickness=2)
                color = tuple(np.random.randint(0,255,3).tolist())
                vis2 = cv2.line(vis2, (int(x0), int(y0)), (int(x1+640), int(y1)), color, 2)
                cv2.imshow('vis', vis2)
                cv2.waitKey(1500)
                vis2 = vis
                cv2.destroyAllWindows()
                
                break
               
            else:
                vis2 = vis
                continue
        vis2 = vis
    cv2.imshow('vis', vis2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
                
            
    
                
            #img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            #img2 = cv2.circle(img2, tuple(pt2), 5,color, -1)
    return vis

### uses H() above to make an x, y list of corners detected, returns Alist and displays all the corners found in green to confirm corners found with H()
def findlist(imagex):
    cv2.imshow('dst',imagex)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    Hlist=H(imagex)
    Alist = [ ]
    rc = -1
    for A in Hlist[0]:
        cc = -1
        rc+=1
        for pix in A:
            cc+=1
            if pix[0] == 0 and pix[1]==0 and pix[2]==255:
                simAlist = [ ]
                simAlist.append(rc)
                simAlist.append(cc)
                Alist.append(simAlist)
                imagex[rc, cc] = [0, 255, 0]
    cv2.imshow('simA', imagex)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return Alist



srcpoints = (np.asarray(findlist(simA)))
srcpoints = np.float32(srcpoints)
dstpoints = np.float32(np.asarray(findlist(simB)))

show = drawlines(simA1, simB1, srcpoints, dstpoints)#aA, bB)
cv2.imshow('simA', show[1])
cv2.imshow('simB', show)
cv2.waitKey(0)
cv2.destroyAllWindows()
np.savetxt('srcpoints.txt',srcpoints, fmt='%d')
np.savetxt('dstpoints.txt',dstpoints, fmt='%d')
print(srcpoints, dstpoints)
Homography = cv2.findHomography(srcpoints, dstpoints, cv2.RANSAC, 10)

print(Homography)
