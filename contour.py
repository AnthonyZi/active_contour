import collections
import numpy as np
#from sklearn.utils.extmath import cartesian
from helper.math import cartesian2

class Contour(object):
    def __init__(self, p_vec, closed=True):
        self.points = np.array(p_vec)
        self.points = PointList(p_vec)
        self.closed = closed

    # sum(distances)**2
    def calc_continuity_energy(self):
        if self.closed:
            tmp_points1 = np.roll(self.points.points, -1, axis=0)
            tmp_points2 = np.array(self.points.points)

        distances = tmp_points2-tmp_points1
        return np.sum( np.square(distances) )

    # sum(laplace(p_i))**2
    def calc_curvature_energy(self):
        if self.closed:
            tmp_points1 = np.roll(self.points.points, -1, axis=0)
            tmp_points2 = np.array(self.points.points)
            tmp_points3 = np.roll(self.points.points, 1, axis=0)

        laplace = tmp_points1 - tmp_points2 + tmp_points3
        return np.sum( np.square(laplace) )

class PointList(object):
    def __init__(self, p_vec):
        self.points = p_vec
        self.boundarys = None

    def set_boundarys(self, boundarys):
        self.boundarys = boundarys

    def get_neighborhood(self,p):
        px,py = p
        p_dx = np.array([px-1,px,px+1])
        p_dy = np.array([py-1, py, py+1])
#        meshgrid_x, meshgrid_y = np.meshgrid(p_dx,p_dy)
#        p_dxdy = np.stack((meshgrid_x, meshgrid_y), axis=-1).reshape(-1,2)
        p_dxdy = cartesian2((p_dx,p_dy,p_dx))
        return p_dxdy


class ActiveContour(Contour):
    def __init__(self, p_vec, boundarys):
        super().__init__(p_vec, True)
        self.points.set_boundarys(boundarys)


if __name__ == "__main__":
    p_vec = [[2,0],[1,1],[2,3],[3,1]]
    print(p_vec)
    c = ActiveContour(p_vec, [0,0,10,10])
    print(c.calc_continuity_energy())
    print(c.calc_curvature_energy())


    p_vec = [[2,0],[1,1],[2,2],[3,1]]
    print(p_vec)
    c = ActiveContour(p_vec, [0,0,10,10])
    print(c.calc_continuity_energy())
    print(c.calc_curvature_energy())


    p_vec = [[1,0],[1,1],[2,2],[3,1]]
    print(p_vec)
    c = ActiveContour(p_vec, [0,0,10,10])
    print(c.calc_continuity_energy())
    print(c.calc_curvature_energy())

    print(c.points.get_neighborhood([1.5,2]))
