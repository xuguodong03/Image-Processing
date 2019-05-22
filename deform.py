import os
import numpy as np
import cv2

def deform(img, grid_size, disturb_degree, mode):

    h,w = img.shape[:2]
    gh, gw = grid_size
    nh = h // gh
    nw = w // gw
    k = disturb_degree

    offset_x = np.zeros((nh+1, nw+1))
    offset_y = np.zeros((nh+1, nw+1))
    offset_x[1:-1, 1:-1] = np.random.randint(low=-int(gw*k-1), high=int(gw*k), size=(nh-1,nw-1), dtype=np.int32)
    offset_y[1:-1, 1:-1] = np.random.randint(low=-int(gh*k-1), high=int(gh*k), size=(nh-1,nw-1), dtype=np.int32)

    grid_x, grid_y = np.meshgrid(np.linspace(0, w, nw+1), np.linspace(0, h, nh+1))
    grid_x = grid_x.round().astype(np.int32)
    grid_y = grid_y.round().astype(np.int32)
    coor_y = grid_y + offset_y
    coor_x = grid_x + offset_x

    grid = np.empty((h,w,2), np.float32)
    mesh_x, mesh_y = np.meshgrid(np.arange(w), np.arange(h))
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
            tmp1 = (mesh_y>=grid_y[i,j]) & (mesh_y<=grid_y[i+1,j])
            tmp2 = (mesh_x>=grid_x[i,j]) & (mesh_x<=grid_x[i,j+1])
            tmp3 = mesh_y-grid_y[i,j] >= mesh_x-grid_x[i,j]
            idx_y, idx_x = np.where(tmp1 & tmp2 & tmp3)
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
            tmp1 = (mesh_y>=grid_y[i,j]) & (mesh_y<=grid_y[i+1,j])
            tmp2 = (mesh_x>=grid_x[i,j]) & (mesh_x<=grid_x[i,j+1])
            tmp3 = mesh_y-grid_y[i,j] <= mesh_x-grid_x[i,j]
            idx_y, idx_x = np.where(tmp1 & tmp2 & tmp3)
            tmp = np.stack((mesh_x[idx_y, idx_x], mesh_y[idx_y, idx_x], np.ones(len(idx_y))))
            tmp = np.dot(affine, tmp)
            grid[idx_y, idx_x, :] = tmp[:2].T

    if mode == 'numpy':
        if np.isnan(grid).any():
            print('twf')
        rx = np.floor(grid[...,0])
        ry = np.floor(grid[...,1])
        x = np.dstack((rx, rx+1, rx, rx+1)).clip(0, w-1).astype(np.int32)
        y = np.dstack((ry, ry, ry+1, ry+1)).clip(0, h-1).astype(np.int32)
        bil_param = np.abs((1-np.abs(grid[...,0:1]-x))*(1-np.abs(grid[...,1:2]-y)))
        I = img.copy().astype(np.float32)
        numpy_img = np.zeros_like(I)
        for i in range(4):
            numpy_img += bil_param[:,:,i:i+1] * I[y[:,:,i], x[:,:,i], :]
        warp_img = numpy_img.astype(np.uint8)

    elif mode == 'torch':
        import torch
        I = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0)
        grid[...,0] = np.clip(grid[...,0], 0, w-1)
        grid[...,1] = np.clip(grid[...,1], 0, h-1)
        grid[...,0] = 2*grid[...,0]/(w-1) - 1
        grid[...,1] = 2*grid[...,1]/(h-1) - 1
        grid = torch.from_numpy(grid).unsqueeze(0)
        warp_img = torch.nn.functional.grid_sample(I, grid)[0].permute(1,2,0).numpy().astype(np.uint8)

    return warp_img

def deform_part(img, box, grid_size, disturb_degree, mode='numpy'):

    I = img.copy()
    y1,y2,x1,x2 = box
    I[y1:y2, x1:x2, :] = deform(img[y1:y2, x1:x2, :], grid_size, disturb_degree, mode)
    return I

if __name__ == '__main__':

    mode = 'numpy'
    # global deform
    grid_size = (20,20)
    disturb = 0.4
    img = cv2.imread('./images/venice.jpg')
    h,w,_ = img.shape
    y1,y2,x1,x2 = 0,h,0,w
    warp_img = deform_part(img, [y1,y2,x1,x2], grid_size, disturb, mode)
    cv2.imwrite('./images/venice_global.jpg', warp_img)
    # local deform
    grid_size = (30,30)
    disturb = 0.5
    img = cv2.imread('./images/cxk.jpg')
    y1,y2,x1,x2 = 150,550,570,970
    warp_img = deform_part(img, [y1,y2,x1,x2], grid_size, disturb, mode)
    cv2.rectangle(warp_img, (x1,y1), (x2,y2), (0,0,192), 2)
    cv2.imwrite('./images/cxk_local.jpg', warp_img)
