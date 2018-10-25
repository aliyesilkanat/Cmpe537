# Generated with SMOP  0.41-beta
from libsmop import *
# HW2\mosaic.m

# load input images
I1=double(imread('left-1.jpg'))
# HW2\mosaic.m:2
h1,w1,d1=size(I1,nargout=3)
# HW2\mosaic.m:3
I2=double(imread('left-2.jpg'))
# HW2\mosaic.m:4
h2,w2,d2=size(I2,nargout=3)
# HW2\mosaic.m:5
# show input images and prompt for correspondences
figure
subplot(1,2,1)
image(I1 / 255)
axis('image')
hold('on')
title('first input image')
X1,Y1=ginput(2,nargout=2)
# HW2\mosaic.m:9

subplot(1,2,2)
image(I2 / 255)
axis('image')
hold('on')
title('second input image')
X2,Y2=ginput(2,nargout=2)
# HW2\mosaic.m:12

# estimate parameter vector (t)
Z=concat([[X2.T,Y2.T],[Y2.T,- X2.T],[1,1,0,0],[0,0,1,1]]).T
# HW2\mosaic.m:14
xp=concat([[X1],[Y1]])
# HW2\mosaic.m:15
t=numpy.linalg.solve(Z,xp)
# HW2\mosaic.m:16

a=t(1)
# HW2\mosaic.m:17

b=t(2)
# HW2\mosaic.m:18

tx=t(3)
# HW2\mosaic.m:19
ty=t(4)
# HW2\mosaic.m:20
# construct transformation matrix (T)
T=concat([[a,b,tx],[- b,a,ty],[0,0,1]])
# HW2\mosaic.m:22
# warp incoming corners to determine the size of the output image (in to out)
cp=dot(T,concat([[1,1,w2,w2],[1,h2,1,h2],[1,1,1,1]]))
# HW2\mosaic.m:24
Xpr=arange(min(concat([cp(1,arange()),0])),max(concat([cp(1,arange()),w1])))
# HW2\mosaic.m:25

Ypr=arange(min(concat([cp(2,arange()),0])),max(concat([cp(2,arange()),h1])))
# HW2\mosaic.m:26

Xp,Yp=ndgrid(Xpr,Ypr,nargout=2)
# HW2\mosaic.m:27
wp,hp=size(Xp,nargout=2)
# HW2\mosaic.m:28

# do backwards transform (from out to in)
X=numpy.linalg.solve(T,concat([ravel(Xp),ravel(Yp),ones(dot(wp,hp),1)]).T)
# HW2\mosaic.m:30

# re-sample pixel values with bilinear interpolation
clear('Ip')
xI=reshape(X(1,arange()),wp,hp).T
# HW2\mosaic.m:33
yI=reshape(X(2,arange()),wp,hp).T
# HW2\mosaic.m:34
Ip[arange(),arange(),1]=interp2(I2(arange(),arange(),1),xI,yI,'*bilinear')
# HW2\mosaic.m:35

Ip[arange(),arange(),2]=interp2(I2(arange(),arange(),2),xI,yI,'*bilinear')
# HW2\mosaic.m:36

Ip[arange(),arange(),3]=interp2(I2(arange(),arange(),3),xI,yI,'*bilinear')
# HW2\mosaic.m:37

# offset and copy original image into the warped image
offset=- round(concat([min(concat([cp(1,arange()),0])),min(concat([cp(2,arange()),0]))]))
# HW2\mosaic.m:39
Ip[arange(1 + offset(2),h1 + offset(2)),arange(1 + offset(1),w1 + offset(1)),arange()]=double(I1(arange(1,h1),arange(1,w1),arange()))
# HW2\mosaic.m:40
# show the result
figure
image(Ip / 255)
axis('image')
title('mosaic image')