#第一宇宙速度を求めるためのシミュレーション サンプルプログラム１
import numpy as np
import matplotlib.pyplot as plt

#定数
mass_s = 100 #人工衛星の質量[kg]
mass_e = 5.972e24 #地球の質量[kg]
R_e = 6356752 #地球の半径 [m]
G = 6.6743e-11 #万有引力定数
h = 100e3 #初期高度
v0 = 7900 #初速 []m/s
deltaT = 0.01 #シミュレーションのきざみ幅 [s]

#初期化
x = 0.0
y = h + R_e
vx = -v0
vy = 0.0
t = 0.0

#表示用変数
T=[]
X=[]
Y=[]

tmax= 3600.0 * 2
while t < tmax:
    r = np.sqrt(x**2 + y**2)
    F = G * mass_s * mass_e / r**2
    xdot = -F * x / r / mass_s
    ydot = -F * y / r / mass_s
    x += vx * deltaT 
    y += vy * deltaT
    vx += xdot * deltaT
    vy += ydot * deltaT
    t += deltaT
    if r < R_e:
        break
    
    T.append(t)
    X.append(x)
    Y.append(y)

theta = np.linspace(0, 2*np.pi, 60)
xe= R_e * np.cos(theta)
ye= R_e * np.sin(theta)

plt.plot(xe,ye)
plt.plot(X,Y)
plt.axis("equal")
plt.grid()
plt.show()
