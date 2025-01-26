#第一宇宙速度を求めるためのシミュレーション サンプルプログラム2
import numpy as np
import matplotlib.pyplot as plt

#Parameters
mass_s = 100 #Mass of artificial satellite [kg]
mass_e = 5.972e24 #Mass of Earth [kg]
R_e = 6356752 #Radius of Earth [m]
G = 6.6743e-11 #Gravitational constant
h = 100e3 #Initial altitude
v0 = 10000 #Initial velocity [m/s]
deltaT = 0.01 #Simulation time step [s]

def state_dot(state):
    #State vector: [x, y, vx, vy]
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    xdot = vx
    ydot = vy
    r = np.sqrt(x**2 + y**2)
    vxdot = - x * G * mass_e / r**3
    vydot = - y * G * mass_e / r**3 
    output = np.array([xdot, ydot, vxdot, vydot])
    return output

def step(func, state, h):
    #Integrate by Euler method
    state = state + func(state) * h
    return state

#Initialization
state = np.array([0.0, h + R_e, -v0, 0.0])
t = 0.0

#Display variables
T=[]
X=[]
Y=[]
VX=[]
VY=[]

#Simulation
tmax= 3600.0 * 6
while t < tmax:
    state = step(state_dot, state, deltaT)
    t += deltaT
    if np.linalg.norm(state[0:2]) < R_e:
        break
    T.append(t)
    X.append(state[0])
    Y.append(state[1])
    VX.append(state[2])
    VY.append(state[3])

#Plot
theta = np.linspace(0, 2*np.pi, 60)
xe= R_e * np.cos(theta)
ye= R_e * np.sin(theta)
plt.plot(xe,ye)
plt.plot(X,Y)
plt.axis('equal')
plt.grid()
plt.show()
