import numpy as np
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
import pandas as pd

rho = 2.7e3
A = 1.2e-4
E = 6.9e10
L = 0.5
I = 9e-11
c1 = 0
c2 = 0
Ne = 100
xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)
propConstant = 1.0 #dependance of force on voltage

# # remove later - just for testing ----- TODO
rho = 1
A = 1
E = 1
L = 1
I = 1
# # ---- remove later ends ----

def getBeamMassAndStiffnessMatrix(rho,A,E,I,Le,n):

    Me = rho*A*Le/420*np.array([[156,    22*Le,   54,     -13*Le],
                                [22*Le,  4*Le**2,  13*Le,  -3*Le**2],
                                [54,     13*Le,   156,    -22*Le],
                                [-13*Le, -3*Le**2, -22*Le, 4*Le**2]])

    Ke = E*I/(Le**3)*np.array([[12,    6*Le,    -12,    6*Le],
                            [6*Le,  4*Le**2,  -6*Le,  2*Le**2],
                            [-12,   -6*Le,   12,     -6*Le],
                            [6*Le,  2*Le**2,  -6*Le,  4*Le**2]])

    Ma = np.zeros([2*n,2*n])
    Ka = np.zeros([2*n,2*n])
    for i in range(0, 2*n-3, 2):
        Ma[i:i+4,i:i+4] = Ma[i:i+4,i:i+4] + Me
        Ka[i:i+4,i:i+4] = Ka[i:i+4,i:i+4] + Ke

    Ma = np.delete(Ma, [0,1], 1)
    Ma = np.delete(Ma, [0,1], 0)
    Ka = np.delete(Ka, [0,1], 1)
    Ka = np.delete(Ka, [0,1], 0)
    return Ma, Ka

[Ma, Ka] = getBeamMassAndStiffnessMatrix(rho, A, E, I, L/Ne, Ne+1)
Ca = (c1*Ma + c2*Ka)

Lambda = 1.875104069/L
# Lambda = 4.694091133/L
# Lambda = 7.854757438/L
# Lambda = 10.99554073/L
# Lambda = 14.13716839/L 
# Lambda = 17.27875953/L

omega = Lambda**2*np.sqrt(E*I/rho/A)
h1 = np.cosh(Lambda*xx) -np.cos(Lambda*xx) -(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
    /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.sinh(Lambda*xx)-np.sin(Lambda*xx))
h2 = Lambda*(np.sinh(Lambda*xx)+np.sin(Lambda*xx))-(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
    /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.cosh(Lambda*xx)-np.cos(Lambda*xx))*Lambda

def nextStateNewmark(M,C,K,voltage,D0,V0,Beta=1/4,Gamma=1/2):
    c1 = 1/Beta/dt**2
    c2 = Gamma/Beta/dt
    c3 = 1/Beta/dt
    c4 = 1/2/Beta-1
    c5 = Gamma/Beta-1
    c6 = (Gamma/2/Beta-1)*dt
    c7 = (1-Gamma)*dt
    c8 = Gamma*dt
    
    A0 = np.dot( np.linalg.inv(M), (- np.dot(K,D0)- np.dot(C,V0)))
    
    Kbar = c1*M + c2*C + K

    F_to_add = np.zeros(D0.shape)
    # F_to_add[19:59:2] = propConstant * voltage
    F_to_add[20:60:2] = propConstant * voltage
    # F_to_add[-2] = propConstant * voltage

    F_load = np.zeros(D0.shape)
    F_load[20:60:2] = np.sin(i)

    Fbar = (F_to_add + F_load +np.dot(M, (c1*D0+c3*V0+c4*A0)) + np.dot(C, (c2*D0+c5*V0+c6*A0)))
    
    D_next = np.matmul(np.linalg.inv(Kbar), Fbar)
    A_next = c1*(D_next-D0) -c3*V0 -c4*A0
    V_next = V0 +c7*A0 +c8*A_next
    return D_next, V_next, A_next

def predict_future_states(D_current, V_current, horizon, control_sequence):
    D_history = []
    V_history = []

    D_history.append(D_current)
    V_history.append(V_current)
    for i in range(horizon):
        [D_next, V_next, _] = nextStateNewmark(Ma, Ca, Ka, control_sequence[i], D_history[-1], V_history[-1])
        D_history.append(D_next)
        V_history.append(V_next)

    return D_history[1:], V_history[1:]

def objective_function(predicted_displacements, predicted_velocities, control_sequence):
    displacement_weight = 1.0
    velocity_weight = 1.0
    control_weight = 0.0
    displacement_cost = sum(np.sum(displacement[0:200:2]**2) for displacement in predicted_displacements)
    velocity_cost = sum(np.sum(velocity[0:200:2]**2) for velocity in predicted_velocities)
    control_cost = sum(voltage**2 for voltage in control_sequence)
    total_cost = displacement_weight*displacement_cost + control_weight*control_cost + velocity_weight * velocity_cost
    return total_cost

def mpc_optimization_problem(D_current, V_current, horizon, objective_function, bounds):
    initial_guess = np.ones(horizon) * 10

    def cost_function(control_sequence):
        [D_predicted, V_predicted] = predict_future_states(D_current, V_current, horizon, control_sequence)
        return objective_function(D_predicted, V_predicted, control_sequence)
    
    result = minimize(cost_function, initial_guess, bounds=bounds)

    return result.x

D0 = np.zeros(2*Ne)
D0[0::2] = h1
D0[1::2] = h2
V0 = np.zeros(2*Ne)
dt = 1e-2
T = 2

plt.plot(np.linspace(0,L,Ne),D0[0::2],label='Beam profile')
plt.xlabel('Distance (m)')
plt.ylabel('Beam displacement (m)')
plt.ylim(-5,5)
plt.title('Initial condition profile')
plt.show()

D_history = []
D = D0
V = V0
horizon = 1
voltage_lower_bound = -1000
voltage_upper_bound = 1000
voltage_bounds = Bounds([voltage_lower_bound] * horizon, [voltage_upper_bound] * horizon)
tracking_node_displacement = []

u_history = []
for i in range(int(T//dt)):
    print('Iteration:',i)
    # optimal_control_sequence = mpc_optimization_problem(D, V, horizon, objective_function, voltage_bounds)
    # [D, V, A] = nextStateNewmark(Ma, Ca, Ka, optimal_control_sequence[0], D, V)
    # print(optimal_control_sequence[0])
    [D, V, A] = nextStateNewmark(Ma, Ca, Ka, 0, D, V)
    D_history.append(D[0::2])
    # u_history.append(optimal_control_sequence[0])
    tracking_node_displacement.append(D[-2])

D = D0
V = V0
tracking_node_displacement2 = []
for i in range(int(T//dt)):
    print('Iteration:',i)
    optimal_control_sequence = mpc_optimization_problem(D, V, horizon, objective_function, voltage_bounds)
    [D, V, A] = nextStateNewmark(Ma, Ca, Ka, optimal_control_sequence[0], D, V)
    # print(optimal_control_sequence[0])
    # [D, V, A] = nextStateNewmark(Ma, Ca, Ka, 0, D, V)
    u_history.append(optimal_control_sequence[0])
    tracking_node_displacement2.append(D[-2])

# D_history = np.array(D_history)
# print(D_history)

# Colorbar plot of all nodes at all times
# fig = plt.figure(figsize=(14,2.5))
# plt.imshow(D_history.T, cmap='jet', origin='lower', aspect='auto'); 
# cbar = plt.colorbar()
# cbar.set_label('Displacement')
# plt.xlabel('Beam Nodes')
# plt.ylabel('Time')
# plt.tight_layout()

# Displacement time graph of a particular node that is tracked
# plt.figure(figsize=(10, 6))
# from scipy.io import loadmat
# data = loadmat('initial_condition_control')
time_points = np.linspace(0,T,int(T//dt))
plt.plot(time_points, tracking_node_displacement2, label='Traced Node Displacement (with control)')
plt.plot(time_points, tracking_node_displacement, label='Traced Node Displacement (without control)')
plt.legend()
plt.grid(True)
plt.xlabel('Time (s)', fontsize = 20)
plt.ylabel('Displacement [m]', fontsize= 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
# plt.ylim(-0.035,0.035)
plt.title('Displacement of the Traced Node Over Time', fontsize = 20)
plt.show()

plt.plot(time_points, u_history, label='Control Voltage')
plt.xlabel('Time (s)', fontsize = 20)
plt.ylabel('Voltage [V]', fontsize = 20)
plt.title('Applied control voltage vs time', fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend()
plt.grid(True)
plt.show()

# fig, ax = plt.subplots()
# nodes = np.linspace(0, L, Ne)
# line, = ax.plot(nodes, D_history[0], 'b-')
# ax.set_xlim(0, L)
# ax.set_ylim(-8, 8)
# ax.set_xlabel('Position along the beam')
# ax.set_ylabel('Displacement')
# ax.set_title('Beam Vibration Over Time')

# def animate(i):
#     line.set_ydata(D_history[i])  # Update the displacement data
#     return line,

# anim = FuncAnimation(fig, animate, frames=len(D_history), interval=1/dt, blit=True)
# plt.show()
# anim.save('beam_vibration.mp4', writer='ffmpeg')