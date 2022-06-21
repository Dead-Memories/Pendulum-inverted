import numpy as np

import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as pp
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from math import pi
from numpy import sin, cos

# physical constants
g = 9.8
l = 1
m = 1
M = 20

# simulation time
dt = 0.05
Tmax = 20
t = np.arange(0.0, Tmax, dt)

# initial conditions
x = 1      # cart position
v = .0      # cart velocity
phi = .0	# pendulum angle
omega = .0 	# pendulum angular velocity

state = np.array([x, v, phi, omega])
def derivatives(state, t):
    ds = np.zeros_like(state)

    ds[0] = state[1]
    ds[1] = (M+m)/(M+m*(sin(state[2]))**2)*(m*g*sin(state[2])*cos(state[2])/(M+m)-m*l*(state[3])**2*sin(state[2])/(M+m))
    ds[2] = state[3]
    ds[3] = (M+m)/(M+m*(sin(state[2]))**2)*(g*sin(state[2])/l-m*(state[3])**2*sin(state[2])*cos(state[2])/(M+m))

    return ds

solution = integrate.odeint(derivatives, state, t)

phis = solution[:, 2]
xs = solution[:, 0]

pxs = l * sin(phis) + xs
pys = l * cos(phis)

fig = pp.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 6), ylim=(-3, 3))
ax.set_aspect('equal')
ax.grid()

patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

cart_width = 0.3
cart_height = 0.2

def init():
    line.set_data([], [])
    time_text.set_text('')
    patch.set_xy((-cart_width/2, -cart_height/2))
    patch.set_width(cart_width)
    patch.set_height(cart_height)
    return line, time_text, patch


def animate(i):
    thisx = [xs[i], pxs[i]]
    thisy = [0, pys[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    patch.set_x(xs[i] - cart_width/2)
    return line, time_text, patch

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(solution)), interval=25, blit=True, init_func=init)

pp.show()

f = r"C:\Users\USER\Desktop\animation.gif"
Writer = animation.writers['imagemagick']
writer = Writer(fps=25, metadata=dict(artist='Olga Khramova'), bitrate=1800)
ani.save(f, writer=writer)