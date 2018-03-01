Active Contours in Python

Initial contour-points are either loaded by an image here black dots or as a point-array

current state (implemented forces):
    f1 = first derivative of pi (curve-points)
    f2 = second derivative of pi (curve-points)
    f3 = pixel-itensity (multiplicative)

    force is calculated:
    (f1+f2) * f3

    -> current state will have a shrinking snake.
    snake shrinks until a white pixel is found

use requirements.txt for dependencies
