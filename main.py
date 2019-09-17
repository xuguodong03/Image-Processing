import os
import numpy as np
import cv2
from skimage import draw

def deform(h, w, nh, nw, degree=0.45, prob=0.5):

    grid_x, grid_y = np.meshgrid(np.linspace(0, w, nw+1), np.linspace(0, h, nh+1))
    grid_x = grid_x.round().astype(np.int32)
    grid_y = grid_y.round().astype(np.int32)
    mesh_x, mesh_y = np.meshgrid(np.arange(w), np.arange(h))

    gh = grid_y[1,0] - grid_y[0,0]
    gw = grid_x[0,1] - grid_x[0,0]

    offset_x = np.zeros((nh+1, nw+1))
    offset_y = np.zeros((nh+1, nw+1))
    offset_x[1:-1, 1:-1] = np.random.randint(low=-int(gw*degree-1), high=int(gw*degree), size=(nh-1,nw-1), dtype=np.int32)
    offset_y[1:-1, 1:-1] = np.random.randint(low=-int(gh*degree-1), high=int(gh*degree), size=(nh-1,nw-1), dtype=np.int32)

    mask = np.random.choice([0,1], size=offset_x.shape, p=[1-prob, prob])
    offset_x *= mask
    offset_y *= mask

    coor_y = grid_y + offset_y
    coor_x = grid_x + offset_x
    grid = np.empty((h,w,2), np.float32)

    for i in range(nh):
        for j in range(nw):
            # lower triangle
            X = np.array([[grid_x[i,j], grid_y[i,j], 1], 
                        [grid_x[i+1,j], grid_y[i+1,j], 1], 
                        [grid_x[i+1,j+1], grid_y[i+1,j+1], 1]]).T
            Y = np.array([[coor_x[i,j], coor_y[i,j], 1], 
                        [coor_x[i+1,j], coor_y[i+1,j], 1],
                        [coor_x[i+1,j+1], coor_y[i+1,j+1], 1]]).T
            affine = np.dot(Y, np.linalg.inv(X))
            idx_y, idx_x = draw.polygon([grid_y[i,j],grid_y[i+1,j],grid_y[i+1,j+1]],
                                        [grid_x[i,j],grid_x[i+1,j],grid_x[i+1,j+1]],
                                        shape=(h,w))
            tmp = np.stack((mesh_x[idx_y, idx_x], mesh_y[idx_y, idx_x], np.ones(len(idx_y))))
            tmp = np.dot(affine, tmp)
            grid[idx_y, idx_x, :] = tmp[:2].T
            # upper triangle
            X = np.array([[grid_x[i,j], grid_y[i,j], 1], 
                        [grid_x[i,j+1], grid_y[i,j+1], 1], 
                        [grid_x[i+1,j+1], grid_y[i+1,j+1], 1]]).T
            Y = np.array([[coor_x[i,j], coor_y[i,j], 1], 
                        [coor_x[i,j+1], coor_y[i,j+1], 1],
                        [coor_x[i+1,j+1], coor_y[i+1,j+1], 1]]).T
            affine = np.dot(Y, np.linalg.inv(X))
            idx_y, idx_x = draw.polygon([grid_y[i,j],grid_y[i,j+1],grid_y[i+1,j+1]],
                                        [grid_x[i,j],grid_x[i,j+1],grid_x[i+1,j+1]],
                                        shape=(h,w))
            tmp = np.stack((mesh_x[idx_y, idx_x], mesh_y[idx_y, idx_x], np.ones(len(idx_y))))
            tmp = np.dot(affine, tmp)
            grid[idx_y, idx_x, :] = tmp[:2].T

    offset = np.stack((offset_x, offset_y), axis=0).astype(np.float32)
    return grid.astype(np.float32), offset[:,1:-1,1:-1].astype(np.float32)

def recover(h, w, nh, nw, offset):

    oc,oh,ow = offset.shape
    assert oh+1==nh and ow+1==nw
    mesh_x, mesh_y = np.meshgrid(np.arange(w),np.arange(h))
    grid_x, grid_y = np.meshgrid(np.linspace(0,w,nw+1), np.linspace(0,h,nh+1))
    deform_x = grid_x.copy()
    deform_x[1:-1,1:-1] += offset[0]
    deform_y = grid_y.copy()
    deform_y[1:-1,1:-1] += offset[1]
    
    affine_pool = []
    index_map = np.zeros((h,w), dtype=np.int32)
    cls = 0 
    grid = np.empty((h,w,2))
    for i in range(nh):
        for j in range(nw):
            Y = np.array([[grid_x[i,j],grid_y[i,j],1],
                           [grid_x[i+1,j],grid_y[i+1,j],1],
                           [grid_x[i+1,j+1],grid_y[i+1,j+1],1]]).T
            X = np.array([[deform_x[i,j],deform_y[i,j],1],
                           [deform_x[i+1,j],deform_y[i+1,j],1],
                           [deform_x[i+1,j+1],deform_y[i+1,j+1],1]]).T
            affine = np.dot(Y, np.linalg.inv(X))
            y,x = draw.polygon(X[1],X[0], shape=(h,w))
            X = np.stack((x,y,np.ones_like(x)))
            Y = np.dot(affine, X)
            grid[y,x,:] = Y[:2].T
    
            Y = np.array([[grid_x[i,j],grid_y[i,j],1],
                           [grid_x[i,j+1],grid_y[i,j+1],1],
                           [grid_x[i+1,j+1],grid_y[i+1,j+1],1]]).T
            X = np.array([[deform_x[i,j],deform_y[i,j],1],
                           [deform_x[i,j+1],deform_y[i,j+1],1],
                           [deform_x[i+1,j+1],deform_y[i+1,j+1],1]]).T
            affine = np.dot(Y, np.linalg.inv(X))
            y,x = draw.polygon(X[1],X[0], shape=(h,w))
            X = np.stack((x,y,np.ones_like(x)))
            Y = np.dot(affine, X)
            grid[y,x,:] = Y[:2].T
    
    return grid.astype(np.float32)

if __name__ == '__main__':

    img = cv2.imread('./images/venice.jpg')
    h,w = img.shape[:2]
    grid, offset = deform(h,w,10,10)
    warp_img = cv2.remap(img, grid[...,0], grid[...,1], interpolation=cv2.INTER_LINEAR)
    grid = recover(h,w,10,10,offset)
    recover_img = cv2.remap(warp_img, grid[...,0], grid[...,1], interpolation=cv2.INTER_LINEAR)

    cv2.imwrite('./images/venice_warp.jpg', warp_img)
    cv2.imwrite('./images/venice_recover.jpg', recover_img)    
