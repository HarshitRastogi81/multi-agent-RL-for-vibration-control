import gymnasium as gym
from gymnasium import spaces
import numpy as np
# rho = 2.7e3, A = 1.2e-4, E = 6.9e10, L = 0.5, I = 9e-11, c1 = 0, c2 = 0, Ne = 100
class BeamEnvironment(gym.Env):
    def __init__(self, rho = 1, A = 1, E = 1, L = 1, I = 1, c1 = 0, c2 = 0, Ne = 100):
        super(BeamEnvironment, self).__init__()
        self.rho = rho
        self.A = A
        self.E = E
        self.L = L
        self.I = I
        self.c1 = c1
        self.c2 = c2
        self.Ne = Ne
        self.xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)
        self.Ma, self.Ka = self.getBeamMassAndStiffnessMatrix(rho, A, E, I, L/Ne, Ne + 1)
        self.Ca = (c1*self.Ma + c2*self.Ka)
        self.Lambda = 1.875104069/L
        self.dt = 1e-2
        # self.Lambda = 4.694091133/self.L
        # self.Lambda = 7.854757438/self.L
        # self.Lambda = 10.99554073/self.L
        # self.Lambda = 14.13716839/self.L
        # self.Lambda = 17.27875953/self.L
        self.displacement_threshold = 0.8
        self.velocity_threshold = 6.0
        self.omega = self.Lambda**2*np.sqrt(E*I/rho/A)
        self.h1 = np.cosh(self.Lambda*self.xx) -np.cos(self.Lambda*self.xx) -(np.cos(self.Lambda*L)+np.cosh(self.Lambda*L)) \
            /(np.sin(self.Lambda*L)+np.sinh(self.Lambda*L))*(np.sinh(self.Lambda*self.xx)-np.sin(self.Lambda*self.xx))
        self.h2 = self.Lambda*(np.sinh(self.Lambda*self.xx)+np.sin(self.Lambda*self.xx))-(np.cos(self.Lambda*L)+np.cosh(self.Lambda*L)) \
            /(np.sin(self.Lambda*L)+np.sinh(self.Lambda*L))*(np.cosh(self.Lambda*self.xx)-np.cos(self.Lambda*self.xx))*self.Lambda
        self.action_limit = 30
        self.action_space = spaces.Box(low=np.array([-self.action_limit]), high=np.array([self.action_limit]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4*Ne,), dtype=np.float32)
        self.reset()

    def nextStateNewmark(self,voltage,D0,V0,Beta=1/4,Gamma=1/2):
        c1 = 1/Beta/self.dt**2
        c2 = Gamma/Beta/self.dt
        c3 = 1/Beta/self.dt
        c4 = 1/2/Beta-1
        c5 = Gamma/Beta-1
        c6 = (Gamma/2/Beta-1)*self.dt
        c7 = (1-Gamma)*self.dt
        c8 = Gamma*self.dt
        
        A0 = np.dot( np.linalg.inv(self.Ma), (- np.dot(self.Ka,D0)- np.dot(self.Ca,V0)))
        
        Kbar = c1*self.Ma + c2*self.Ca + self.Ka

        F_to_add = np.zeros(D0.shape)
        F_to_add[20:60:2] = 1.0 * voltage

        F_load = np.zeros(D0.shape)
        # F_load[20:60:2] = np.sin(i)

        Fbar = (F_to_add + F_load +np.dot(self.Ma, (c1*D0+c3*V0+c4*A0)) + np.dot(self.Ca, (c2*D0+c5*V0+c6*A0)))
        
        D_next = np.matmul(np.linalg.inv(Kbar), Fbar)
        A_next = c1*(D_next-D0) -c3*V0 -c4*A0
        V_next = V0 +c7*A0 +c8*A_next
        return D_next, V_next
    
    def getBeamMassAndStiffnessMatrix(self,rho,A,E,I,Le,n):
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
    
    def reset(self):
        self.D = np.zeros(2*self.Ne)
        self.D[0::2] = self.h1
        self.D[1::2] = self.h2
        self.V = np.zeros(2*self.Ne)
        return np.concatenate([self.D, self.V])
    
    def step(self, action):
        voltage = action[0]
        [D_next, V_next] = self.nextStateNewmark(voltage, self.D, self.V)
        self.D, self.V = D_next, V_next
        state = np.concatenate([self.D, self.V])
        reward = self.calculate_reward(self.D, self.V)
        done = self.isdone(self.D, self.V)
        return state, reward, done, {'D': self.D, 'V':self.V}
    
    def calculate_reward(self, D, V):
        if np.max(np.abs(D[0::2])) > 5 or np.max(np.abs(V[0::2]))>70:
            return -100000
        elif np.all(np.abs(D[0::2]) < self.displacement_threshold) and np.all(np.abs(V[0::2]) < self.velocity_threshold):
            reward = 100000
        else:
            reward = -np.sum([(5*np.max(np.abs(D[0::2]*10)))**2,(5*np.max(np.abs(V[0::2])))**2])/1000
            # reward = -(np.max(np.abs(V[0::2]))**2)/10
        return reward
    
    def isdone(self, D, V):
        done = np.max(np.abs(D[0::2]))>5 or np.max(np.abs(V[0::2]))>70 or (np.all(np.abs(D[0::2]) < self.displacement_threshold) and np.all(np.abs(V[0::2]) < self.velocity_threshold))
        return done