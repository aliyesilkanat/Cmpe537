MAX_KEYPOINTS=100
def calculate_new_points(H12,sift1):
    h12_sift1=np.dot(H12,np.column_stack((sift1,np.ones(sift1.shape[0]))).T)
    y_h12=h12_sift1[0,:]/h12_sift1[2,:]
    x_h12=h12_sift1[1,:]/h12_sift1[2,:]
    return np.vstack((y_h12,x_h12)).T.astype(np.int64)
    
def calc_repeatability(sift,sift2):
    if len(sift)>len(sift2):
        n1=sift
        n2=sift2
    else:
        n1=sift2
        n2=sift
    offset=4
    count=0
    matches=set()
    for (x2,y2),idx2 in zip(n2,range(len(n2))):
        for (x1,y1),idx1 in zip(n1,range(len(n1))):
            if x2+offset>x1 and y2+offset>y1 and x2-offset<x1 and y2-offset<y1:
                if idx1 not in matches:
                    matches.add(idx1)
                    count+=1
    return count/len(n2)

def calc_harris(img):
    gray=img
    block_size = 2
    aperture = 3
    free_parameter = 0.04
    dst = cv2.cornerHarris(gray, block_size, aperture, free_parameter)
   
    res= getharrispoints(dst)
    return res

def calc_sift(img):
    gray=img

    sift = cv2.xfeatures2d.SIFT_create()
    kp,descs = sift.detectAndCompute(gray,None)
    res=np.array([a.response for a in kp])
    kp=np.array(kp)[np.max(res)*0.5<res]
    if len(kp)>MAX_KEYPOINTS:
        kp=kp[:MAX_KEYPOINTS]
    
    return np.array([[p.pt[1],p.pt[0]] for p in kp]).astype(np.int64)



def compute_harris_response(im,sigma=0.5):

    # derivatives	
    imx=np.zeros(im.shape)	
    filters.gaussian_filter(im,	(sigma,	sigma),(0,1),	imx)
    imy=np.zeros(im.shape)			
    filters.gaussian_filter(im,	(sigma,	sigma),(1,0),	imy)

    # compute components of the Harris matrix
    Wxx=filters.gaussian_filter(imx*imx,sigma)
    Wxy=filters.gaussian_filter(imx*imy,sigma)
    Wyy=filters.gaussian_filter(imy*imy,sigma)
    # determinant and trace
    Wdet=Wxx*Wyy-Wxy**2
    Wtr=Wxx+Wyy
    harrisim=Wdet/Wtr
    return getharrispoints(harrisim)


def getharrispoints(harrisim,threshold=0.04,min_dist=10):

    maxnum=0
    harrisim=np.nan_to_num(harrisim)
    maxnum=np.max(harrisim)
    
    cornerthreshold=maxnum*threshold
    harrisimt=(harrisim>cornerthreshold)*1
    coords=np.array(harrisimt.nonzero()).T
    candidatevalues=[harrisim[c[0],c[1]]for c in coords]
    index=np.argsort(candidatevalues)
    allowedlocations=np.zeros(harrisim.shape)
    allowedlocations[min_dist:-min_dist,min_dist:-min_dist]=1
    filteredcoords=[]
    for i in index:
        if allowedlocations[coords[i,0],coords[i,1]]==1:
            filteredcoords.append(coords[i])
            allowedlocations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)]=0
        
        if len(filteredcoords)==MAX_KEYPOINTS:
            break
    return np.array(filteredcoords)