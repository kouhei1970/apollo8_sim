#第一宇宙速度を求めるためのシミュレーション サンプルプログラム3
import numpy as np
import matplotlib.pyplot as plt


class star:
    def __init__(self, mass, radius):
        self.mass = mass
        self.radius = radius
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
    
    def set_state(self, state):
        self.state = state


class system:
    def __init__(self):
        self.stars = []
        self.state = np.array([])

    def add_star(self, star):
        self.stars.append(star)

    def init_state(self):
        self.state = np.array([])
        for star in self.stars:
            self.state = np.append(self.state, star.state)
        return self.state

    def state_dot(self):
        G = 6.6743e-11 #Gravitational constan
        output = np.array([]) 
        for star in self.stars:
            output = np.append(output, star.state[2:4])
            for star2 in self.stars:
                if star == star2:
                    continue
                x = star.state[0] - star2.state[0]
                y = star.state[1] - star2.state[1]
                r = np.sqrt(x**2 + y**2)
                F = G * star.mass * star2.mass / r**2
                vxdot = -F * x / r / star.mass
                vydot = -F * y / r / star.mass
            output = np.append(output, np.array([vxdot, vydot]))
        return output
        
    def step(self, h):
        self.state = self.state + self.state_dot() * h
        index =0
        for star in self.stars:
            star.set_state(self.state[index:index+4])
            index += 4
        return self.state


#Parameters
h = 100e3 #Initial altitude
v0 = 1023 #Initial velocity [m/s]
R_e = 6356752 #Radius of Earth [m]
R_m = 1737100 #Radius of Moon orbit [m]
R_m_orbit = 384400e3 #Radius of Moon orbit [m]
mass_s = 100 #Mass of artificial satellite [kg]
mass_e = 5.972e24 #Mass of Earth [kg]
mass_m = 7.342e22 #Mass of Moon [kg]
deltaT = 1.0 #Simulation time step [s]

satllite = star(mass_m, 0)
earth = star(mass_e, R_e)
satllite.set_state(np.array([0.0, R_m_orbit, -v0, 0.0]))
earth.set_state(np.array([0.0, 0.0, 0.0, 0.0]))

system = system()
system.add_star(satllite)
system.add_star(earth)
system.init_state()

#Display variables
T=[]
X=[]
Y=[]
Xe=[]
Ye=[]
VX=[]
VY=[]

#Simulation
t = 0.0
tmax= 3600*24*27.3*1.5
while t < tmax:
    system.step(deltaT)
    t += deltaT
    if np.linalg.norm(system.stars[0].state[0:2]) < system.stars[1].radius:
        break
    T.append(t)
    X.append(system.stars[0].state[0])
    Y.append(system.stars[0].state[1])
    Xe.append(system.stars[1].state[0])
    Ye.append(system.stars[1].state[1])
    VX.append(system.stars[0].state[2])
    VY.append(system.stars[0].state[3])

#Plot
theta = np.linspace(0, 2*np.pi, 60)
xe= earth.radius * np.cos(theta)
ye= earth.radius * np.sin(theta)
xm= R_m * np.cos(theta)
ym= R_m * np.sin(theta)+R_m_orbit
plt.plot(xe,ye)
plt.plot(xm,ym)
plt.plot(X,Y)
plt.plot(Xe,Ye)
plt.axis("equal")
plt.grid()
plt.show()
