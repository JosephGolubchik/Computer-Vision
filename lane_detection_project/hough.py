import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 8]

def f(x,pi):
    return (-pi[0]/(pi[1]+0.01))*(x-pi[0])+pi[1]

def HoughLines(edges):
#     edges = cv2.Canny(img,100,200)
    h,w = edges.shape[:2]
    max_dist = int(np.sqrt(h**2+w**2))
    theta_inc = 1
    hspace = np.zeros((180//theta_inc+1,max_dist*2+1))
    for i in range(h):
        for j in range(w):
            if edges[i,j] == 255:
                for theta in range(0,181,theta_inc):
                    rtheta = np.deg2rad(theta)
                    d = int(j*np.cos(rtheta)+i*np.sin(rtheta))
                    hspace[theta//theta_inc,d+max_dist] += 1
                    
    lines = np.zeros((edges.shape[0],edges.shape[1],3))
    thresh = 0.7*hspace.max()
    for i in range(hspace.shape[0]):
        for j in range(hspace.shape[1]):
            if hspace[i,j] >= thresh:
                best_theta = np.deg2rad(i)
                best_d = j-max_dist
                pi = (int(best_d*np.sin(np.pi/2-best_theta)),int(best_d*np.cos(np.pi/2-best_theta)))
                pa = (0,int(f(0,pi)))
                pb = (edges.shape[1]-1,int(f(edges.shape[1]-1,pi)))
                lines = cv2.line(lines,pa,pb,(0.6,0,0),2)

    res = np.zeros((edges.shape[0],edges.shape[1],3))
    res[:,:,0] = edges
    res[:,:,1] = edges
    res[:,:,2] = edges
    alpha = 0.6
    res = alpha*lines+(1-alpha)*res
    
    return res, cv2.resize(hspace,(hspace.shape[1],hspace.shape[1]))