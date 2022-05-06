# Formulas to get the frustum

Input variables:

- LiDAR origin $= (x_{0l},y_{0l},z_{0l}) = (0,0,0)$
- Point on image pixels $= (x_{1i},y_{1i})$
- Object size 2D $= (h_i,w_i)$
- Approximate 3D distance $= d$
- Projection matrix $= P_2$
- Rectification matrix $= R_0$
- Velodyne to camera transformation $= T_{v2c}$
- Image size $= (d_x,d_y)$

Steps:

1. Project $(x_{1i},y_{1i})$ to the LiDAR 3D word:

    - Pointcloud to camera: $cam = P_2 * R_0 * T_{v2c} * velo$
    - Solve using the distance in the equation:

        - Point on camera coordinates $= (x_{1c},y_{1c},z_{1c})$

        - $\left(\begin{array}{cc}wx_{1i}\\wy_{1i}\\w\end{array}\right) =
        P_2 \left(\begin{array}{cc}x_{1c}\\y_{1c}\\z_{1c}\\1\end{array}\right) = 
        \left(\begin{array}{cc}f&0&u_0&t_x\\0&f&v_0&t_y\\0&0&1&t_z\end{array}\right)
        \left(\begin{array}{cc}x_{1c}\\y_{1c}\\z_{1c}\\1\end{array}\right)$
        - Equations to obtain $w$:
            - $wx_{1i} = f x_{1c} + u_0 z_{1c} + t_x$
            - $wy_{1i} = f y_{1c} + v_0 z_{1c} + t_y$
            - $w = z_{1c} + t_z$
            - $d = \sqrt{x_{1c}^2 + y_{1c}^2 + z_{1c}^2}$
        - Use of the KITTI camera projection matrix values and the approximate distance to obtain $w$ in each specific case.
            - Values of $P_2$:
                - $f = 707.0493$
                - $u_0 = 604.0814$
                - $v_0 = 180.5066$
                - $t_x = 45.75831$
                - $t_y = -0.3454157$
                - $t_z = 0.004981016$
            - New equations to obtain $w$ or directly $(x_{1c},y_{1c},z_{1c})$:
                - $w x_{1i}=707.0493 x_{1c}+604.0814 z_{1c}+45.75831$
                - $w y_{1i} = 707.0493 y_{1c} + 180.5066 z_{1c} + -0.3454157$
                - $w = z_{1c} + 0.004981016$
                - $d = \sqrt{x_{1c}^2 + y_{1c}^2 + z_{1c}^2}$
        - Use of an approximate variable clearance method due to the complexity of this resolution,


    - $\left(\begin{array}{cc}x_{1l}\\y_{1l}\\z_{1l}\\1\end{array}\right) = 
    T_{v2c}^{-1} * R_0^{-1} * 
    \left(\begin{array}{cc}x_{1l}\\y_{1l}\\z_{1l}\\1\end{array}\right)$

2. Create a 3D line from $(x_{0l},y_{0l},z_{0l})$ to $(x_{1l},y_{1l},z_{1l})$:

    - $l_0$:
        - $x_{l0} = x_{1l}t$
        - $y_{l0} = y_{1l}t$
        - $z_{l0} = z_{1l}t$

<!-- 3. Create perpendicular plane to $l_0$ that passes through $(x_1,y_1,z_1)$:

    - $plane_{p}: x_1(x_{plane}-x_1) + y_1(y_{plane}-y_1) + z_1(z_{plane}-z_1) = 0$ -->

4. Get the 3D points of each vertex for each object and obtain the lines passing through those points and the LiDAR..

5. Create 3D points for each vertex of the cuboid (frustum):

\
\
\
\
\
\

- Transformación de la nube de puntos a coordenadas de la cámra rectificada (como los GT de KITTI)
- Proyección de la nube de puntos a la cámara en 2D
- Creación de una máscara de los puntos en 2D en función de cada detección 2D