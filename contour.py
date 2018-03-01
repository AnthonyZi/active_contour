import collections
import numpy as np
from helper.math import cartesian2
import scipy.ndimage as scipynd
import skimage.io as skiio
import skimage.color as skic
import matplotlib.pyplot as plt


class Contour(object):
    def __init__(self, contour_image=None, contour_points=None, closed=True):
        if contour_image is not None:
            self.points = PointList(self.load_contour_from_image(contour_image))
        elif contour_points is not None:
            self.points = PointList(contour_points)
        self.closed = closed

    def load_contour_from_image(self, image):
        return np.roll(np.stack(np.where(image==1), axis=-1), 1, axis=-1)

    def draw_contour(self,ax):
        c = np.append(self.points.points,self.points.points[0])
        ax.plot(np.real(c),np.imag(c))



class PointList(object):
    def __init__(self, p_vec):
        self.points = np.array(p_vec[:,0] + 1j*p_vec[:,1])
        self.boundarys = [np.iinfo(np.int32).min,np.iinfo(np.int32).max]*2
        self.sort_points()

    # xl,xr,yt,yb
    def set_boundarys(self, boundarys):
        self.boundarys = boundarys

    def move_points(self, offset):
        self.points = self.points + offset

    def sort_points(self):
        tmp_points = self.points - self.points.mean()

        pxy_angles = np.angle(tmp_points)
        ind_sort = np.argsort(pxy_angles)
        sorted_points = self.points[ind_sort]
        self.points = sorted_points



class ActiveContour(Contour):
    def __init__(self, image, contour_image=None, contour_points=None, boundarys=None):
        super().__init__(contour_image, contour_points, True)
        if boundarys != None:
            self.points.set_boundarys(boundarys)
        self.set_image(image)

        self.alpha_c1force = 1
        self.alpha_c2force = 8
        self.alpha_imageforce = 1

        self.calc_forces()

    def set_image(self, image):
        self.image = image
        minval, maxval = image.min(), image.max()
        image_norm = (image-minval)/(maxval-minval)
        sx = scipynd.sobel(image_norm, axis=1, mode="constant")
        sy = scipynd.sobel(image_norm, axis=0, mode="constant")
        self.image_gradient = 1* (sx+1j*sy)

### forces ###
    def calc_forces(self):
        self.calc_c1_forces()
        self.calc_c2_forces()
        self.calc_image_forces()
        self.calc_total_forces()

    # sum(distances)**2
    def calc_c1_forces(self):
        if self.closed:
            tmp_points1 = np.roll(self.points.points, 1)
            tmp_points2 = np.roll(self.points.points, -1)

        central_differences = (tmp_points2-tmp_points1)/2
        #rotate vector to point inside circle
        self.c1forces = self.alpha_c1force * central_differences * np.exp(1j*np.radians(90))

    # sum(laplace(p_i))**2
    def calc_c2_forces(self):
        if self.closed:
            tmp_points1 = np.roll(self.points.points, 1, axis=0)
            tmp_points2 = np.array(self.points.points)
            tmp_points3 = np.roll(self.points.points, -1, axis=0)

        laplaces = tmp_points1 - 2*tmp_points2 + tmp_points3
        self.c2forces = self.alpha_c2force * laplaces

    def calc_image_forces(self):
        p = self.points.points
        px,py = np.real(p).astype(int),np.imag(p).astype(int)
#        self.imageforces = self.alpha_imageforce * self.image_gradient[py,px]
        self.imageforces = 1-self.image[py,px]

    def calc_total_forces(self):
        self.totalforces = (self.c1forces + self.c2forces) * self.imageforces


### iteration ###
    def iteration_step(self):
        self.points.move_points(0.02 * self.totalforces)
        self.calc_forces()



### drawings ###
    def draw_c1_forces(self,ax):
        p = self.points.points
        f = self.c1forces
        ax.quiver(np.real(p),np.imag(p),np.real(f),np.imag(f), color='g')

    def draw_c2_forces(self,ax):
        p = self.points.points
        f = self.c2forces
        ax.quiver(np.real(p),np.imag(p),np.real(f),np.imag(f), color='y')

    def draw_image_forces(self,ax):
        p = self.points.points
        f = self.imageforces
        ax.quiver(np.real(p),np.imag(p),np.real(f),np.imag(f), color='cyan')

    def draw_total_forces(self,ax):
        p = self.points.points
        f = self.totalforces
        ax.quiver(np.real(p),np.imag(p),np.real(f),np.imag(f), color='r')


if __name__ == "__main__":
    img = skic.rgb2gray(skiio.imread("test_neg2.png"))
    contour = np.where(skic.rgb2gray(skiio.imread("test_curve3.png")) <= 0.05, 1, 0)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    c = ActiveContour(image=img, contour_image=contour, boundarys=[0,0,10,10])
    c.draw_contour(ax)
    for i in range(10000):
        c.iteration_step()
        if i%5 == 0:
            c.draw_contour(ax)
#    c.draw_total_forces(ax)
    c.draw_contour(ax)
    plt.show()

#    ax.invert_yaxis()
#    ax.xaxis.tick_top()
