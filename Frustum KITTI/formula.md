# Formulas to get the frustum

Input variables:

- LiDAR origin $= (x_0,y_0,z_0) = (0,0,0)$
- Point on image pixels $= (x_{1i},y_{1i})$
- Object size 2D $= (h_i,w_i)$
- Approximate 3D distance $= d$
- Projection matrix $= P_2$
- focal distance $= f$
- Rectification matrix $= R_0$
- Velodyne to camera transformation $= T_{v2c}$

Steps:

1. Project $(x_{1i},y_{1i})$ to the 3D word:

    - Pointcloud to camera: $cam = P_2 * R_0 * T_{v2c} * velo$
    - Solve the use of disance in the equation:

        - Point on camera coordinates $= (x_{1c},y_{1c},z_{1c})$
        - $x_{1c} f = w x_{1i}$
        - $y_{1c} f = wy_{1i}$
        - $z_{1c} f = w$
        - $d = \sqrt{x_{1c}^2+y_{1c}^2+z_{1c}^2)} = \sqrt{\frac{w²x_{1i}²}{f²}+\frac{w²y_{1i}²}{f²}+w²}$
        - $d = \sqrt{\frac{w²x_{1i}²+w²y_{1i}²+f²w²}{f²}} = \frac{d\sqrt{x_{1i}²+y_{1i}²+f²}}{f}$
        - $w = \frac{df}{\sqrt{x_{1i}²+y_{1i}²+f²}}$

    - $\left(\begin{array}{cc}x_{1}\\y_{1}\\z_{1}\\1\end{array}\right) = 
    T_{v2c}^{-1} * R_0^{-1} * P_2^{-1} *
    \left(\begin{array}{cc}x_{1i}w\\y_{1i}w\\w\end{array}\right)$

2. Create a 3D line from $(x_0,y_0,z_0)$ to $(x_1,y_1,z_1)$:

    - $l_0$:
        - $x_{l0} = x_1t$
        - $y_{l0} = y_1t$
        - $z_{l0} = z_1t$

3. Create perpendicular plane to $l_0$ that passes through $(x_1,y_1,z_1)$:

    - $plane_{p}: x_1(x_{plane}-x_1) + y_1(y_{plane}-y_1) + z_1(z_{plane}-z_1) = 0$

4. Create 3D points for each vertex of the 2D object:

    - $\left(\begin{array}{cc}x_{tl}&x_{tr}&x_{bl}&x_{br}\\
    y_{tl}&y_{tr}&y_{bl}&y_{br}\\
    z_{tl}&z_{tr}&z_{bl}&z_{br}\\
    1&1&1&1\end{array}\right) = 
    T_{v2c}^{-1} * R_0^{-1} * P_2^{-1} *
    (w \left(\begin{array}{cc}x_{tli}&x_{tri}&x_{bli}&x_{bri}\\
    y_{tli}&y_{tri}&y_{bli}&y_{bri}\\
    1&1&1&1\end{array}\right))$

    <!-- - $\left(\begin{array}{cc}l_{tl}\\l_{tr}\\l_{bl}\\l_{br}\end{array}\right)
    = (\left(\begin{array}{cc}x_{tl}&y_{tl}&z_{tl}\\
    x_{tr}&y_{tr}&z_{tr}\\
    x_{bl}&y_{bl}&z_{bl}\\
    x_{br}&y_{br}&z_{br}\end{array}\right) -
    \left(\begin{array}{cc}x_{0}&y_{0}&z_{0}\\
    x_{0}&y_{0}&z_{0}\\
    x_{0}&y_{0}&z_{0}\\
    x_{0}&y_{0}&z_{0}\end{array}\right)) *
    \left(\begin{array}{cc}i\\j\\k\end{array}\right)$ -->

5. Get the 3D points of the intersections of the 3D lines in the plane:

    - Get only one point:
        - $x_1(x_{tl}t-x_1) + y_1(y_{tl}t-y_1) + z_1(z_{tl}t-z_1) = 0$
        - $(x_1 x_{tl}+y_1 y_{tl}+z_1 z_{tl})t + x_1^2 y_1^2 z_1^2 = 0$
        - $t = -\frac{x_1^2 y_1^2 z_1^2}{x_1 x_{tl}+y_1 y_{tl}+z_1 z_{tl}}$
        - Get two points:
            - $x_{tlp} = x_1(x_{tl}t-x_1) + y_1(y_{tl}t-y_1) + z_1(z_{tl}t-z_1) = 0$
    - $\left(\begin{array}{cc}x_{tl}&y_{tl}&z_{tl}\end{array}\right) = 
    \left(\begin{array}{cc}x_1&y_1&z_1\end{array}\right)$

6. Get the points of the rectangle:

    - 