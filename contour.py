import collections
import numpy as np
from helper.math import cartesian2
import scipy.ndimage as scipynd
import skimage.io as skiio

class Contour(object):
    def __init__(self, p_vec=None, image=None, closed=True):
        if image != None:
            self.points = PointList(self.load_contour_from_image(image))
        elif p_vec != None:
            self.points = PointList(p_vec)
        self.closed = closed

    def load_contour_from_image(self, filename):




class PointList(object):
    def __init__(self, p_vec):
        self.points = np.array(p_vec[:,0] + 1j*p_vec[:,1])
        self.boundarys = [np.iinfo(np.int32).min,np.iinfo(np.int32).max]*2

        self.sort_points()

    # xl,xr,yt,yb
    def set_boundarys(self, boundarys):
        self.boundarys = boundarys

#    def get_neighbourhood_indices(self,p):
#        px,py = p
#        p_dx = np.clip(np.array([px-1,px,px+1]), self.boundarys[0], self.boundarys[1])
#        p_dy = np.clip(np.array([py-1,py,py+1]), self.boundarys[2], self.boundarys[3])
##        meshgrid_x, meshgrid_y = np.meshgrid(p_dx,p_dy)
##        p_dxdy = np.stack((meshgrid_x, meshgrid_y), axis=-1).reshape(-1,2)
#        p_dxdy = cartesian2((p_dx,p_dy))
#        return p_dxdy

    def sort_points(self):
        tmp_points = self.points - self.points.mean()

        pxy_angles = np.angle(tmp_points)
        ind_sort = np.argsort(pxy_angles)
        sorted_points = self.points[ind_sort]
        self.points = sorted_points



class ActiveContour(Contour):
    def __init__(self, image, p_vec, boundarys):
        super().__init__(p_vec, True)
        self.points.set_boundarys(boundarys)
        self.set_image(image)

    def set_image(self, image):
        self.image = image
        sx = scipynd.sobel(self.image, axis=0, mode="constant")
        sy = scipynd.sobel(self.image, axis=1, mode="constant")
        self.gradient = np.hypot(sx,sy)


    # sum(distances)**2
    def calc_continuity_force(self):
        if self.closed:
            tmp_points1 = np.roll(self.points.points, -1, axis=0)
            tmp_points2 = np.roll(self.points.points, 1, axis=0)

        central_differences = (tmp_points2-tmp_points1)/2
        #rotate vector to point inside circle
        forces = central_difference*np.exp(1j*np.radians(90))
        return forces

    # sum(laplace(p_i))**2
    def calc_curvature_force(self):
        if self.closed:
            tmp_points1 = np.roll(self.points.points, -1, axis=0)
            tmp_points2 = np.array(self.points.points)
            tmp_points3 = np.roll(self.points.points, 1, axis=0)

        laplaces = tmp_points1 - tmp_points2 + tmp_points3
#        return np.sum( np.square(laplace) )


if __name__ == "__main__":
    img = skiio.imread("55569_positive_000_he_layer0_044334x_017835y.png")

    p_vec = [[2,0],[1,1],[2,3],[3,1]]
    print(p_vec)
    c = ActiveContour(img, p_vec, [0,0,10,10])
    print(c.calc_continuity_force())
    print(c.calc_curvature_force())


    p_vec = [[2,0],[1,1],[2,2],[3,1]]
    print(p_vec)
    c = ActiveContour(img, p_vec, [0,0,10,10])
    print(c.calc_continuity_force())
    print(c.calc_curvature_force())


    p_vec = [[1,0],[1,1],[2,2],[3,1]]
    print(p_vec)
    c = ActiveContour(img, p_vec, [0,0,10,10])
    print(c.calc_continuity_force())
    print(c.calc_curvature_force())

#    print(c.points.get_neighbourhood_indices([1,2]))
#    print(c.points.get_all_neighbourhood_combinations().shape)
