import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Hill functions
def H_plus(Si, T, n):
    return (Si / T) ** n / (1 + (Si / T) ** n)

def H_minus(Si, T, n):
    return 1 / (1 + (Si / T) ** n)

# Kisspeptin
def f_kisspeptin(A, E, f0, T_fA, n_fA, T_fE, n_fE):
    return f0 * H_minus(A, T_fA, n_fA) * H_minus(E, T_fE, n_fE)

def a_kisspeptin(E, a0, T_aEp, n_aEp, T_aEm, n_aEm):
    return a0 * (H_plus(E, T_aEp, n_aEp) + H_minus(E, T_aEm, n_aEm))

# LH
def dLHp_dt(t, LHp, sLH, rLH):
    return sLH - rLH

def sLH(E, A, f, bLHs, T_LHE, nLHE, T_LHA, nLHA, b0, bk, bh):
    return (bLHs * H_plus(E, T_LHE, nLHE) * H_minus(A, T_LHA, nLHA) * bh *
            np.exp(-((f - b0) ** 2) / (bk ** 2)))

def rLH(kLHr, LHp):
    return kLHr * LHp

def dLHb_dt(t, LHb, rLH, kLHcl, Vb):
    return (1 / Vb) * rLH - kLHcl * LHb

# FSH
def dFSHp_dt(t, FSHp, sFSH, rFSH):
    return sFSH - rFSH

def sFSH(A, f, bFSHs, T_FSHA, nFSHA, T_FSHF, nFSHF):
    return bFSHs * H_minus(A, T_FSHA, nFSHA) * H_minus(f, T_FSHF, nFSHF)

def rFSH(kFSHr, FSHp):
    return kFSHr * FSHp

def dFSHb_dt(t, FSHb, rFSH, kFSHcl, Vb):
    return (1 / Vb) * rFSH - kFSHcl * FSHb

# Estrogens
def dEov_dt(t, Eov, sE, rE):
    return sE - rE

def sE(FSHp, A, E0, Emax, T_EFSH, n_EFSH, T_EA, n_EA):
    return E0 + Emax * H_plus(FSHp, T_EFSH, n_EFSH) * H_plus(A, T_EA, n_EA)

def rE(kEr, Eov):
    return kEr * Eov

def dEb_dt(t, Eb, rE, kEcl, Vb):
    return (1 / Vb) * rE - kEcl * Eb

# Androgens
def dAov_dt(t, Aov, sA, rA):
    return sA - rA

def sA(LHp, A0, Amax, T_ALH, n_ALH):
    return A0 + Amax * H_plus(LHp, T_ALH, n_ALH)

def rA(kAr, Aov):
    return kAr * Aov

def dAb_dt(t, Ab, rA, kAcl, Vb):
    return (1 / Vb) * rA - kAcl * Ab

class HormonalDynamicsModel:
    def __init__(self):
        # Default parameter values
        self.parameters = {
            'f0': 1, 'a0': 1, 'T_fA': 1, 'n_fA': 1, 'T_fE': 1, 'n_fE': 1,
            'T_aEp': 1, 'n_aEp': 1, 'T_aEm': 1, 'n_aEm': 1, 'bLHs': 1, 'T_LHE': 1, 'nLHE': 1,
            'T_LHA': 1, 'nLHA': 1, 'b0': 1, 'bk': 1, 'bh': 1,'kLHr': 1,
            'kLHcl': 1, 'Vb': 1, 'bFSHs': 1, 'T_FSHA': 1, 'nFSHA': 1, 'T_FSHF': 1,
            'nFSHF': 1, 'kFSHr': 1, 'kFSHcl': 1, 'E0': 0, 'Emax': 1, 'T_EFSH': 1,
            'n_EFSH': 1, 'T_EA': 1, 'n_EA': 1, 'kEr': 1, 'kEcl': 1, 'A0': 0,
            'Amax': 1, 'T_ALH': 1, 'n_ALH': 1, 'kAr': 1, 'kAcl': 1
        }

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                raise KeyError(f"Parameter '{key}' is not recognized.")
    
    def f_kisspeptin(self, A, E):
        p = self.parameters
        return f_kisspeptin(A, E, p['f0'], p['T_fA'], p['n_fA'], p['T_fE'], p['n_fE'])
    
    def a_kisspeptin(self, E):
        p = self.parameters
        return a_kisspeptin(E, p['a0'], p['T_aEp'], p['n_aEp'], p['T_aEm'], p['n_aEm'])

# Сreating the model and setting parameters
model = HormonalDynamicsModel()
model.set_parameters(f0=16,a0=0.0056,T_fA=3,n_fA=2,T_fE=120,n_fE=10,T_aEp=100,n_aEp=2,T_aEm=9.6,n_aEm=1,
                   bLHs=1827.48,T_LHE=192.2,nLHE=5,T_LHA=2.371,nLHA=1,b0=1,bk=1,bh=1,kLHr=10,kLHcl=74.851,Vb=5,bFSHs=16000,
                   T_FSHA=2,nFSHA=2,T_FSHF=15,nFSHF=5,kFSHr=0.02,kFSHcl=114.25,E0=0,Emax=1,T_EFSH=1,n_EFSH=1,T_EA=1,
                   n_EA=1,kEr=1,kEcl=1,A0=0,Amax=1,T_ALH=1,n_ALH=1,kAr=1,kAcl=1)

def model_odes(t, y, params):
    # Unpack parameters and state variables
    Eov, Aov, LHp, LHb, FSHp, FSHb, Eb, Ab = y
    p = params

    # Define the ODEs
    dEov_dt = sE(FSHp, Aov, p['E0'], p['Emax'], p['T_EFSH'], p['n_EFSH'], p['T_EA'], p['n_EA']) - rE(p['kEr'], Eov)
    dAov_dt = sA(LHp, p['A0'], p['Amax'], p['T_ALH'], p['n_ALH']) - rA(p['kAr'], Aov)
    dLHp_dt = sLH(Eb, Ab, f_kisspeptin(Aov, Eb, p['f0'], p['T_fA'], p['n_fA'], p['T_fE'], p['n_fE']),
                  p['bLHs'], p['T_LHE'], p['nLHE'], p['T_LHA'], p['nLHA'], p['b0'], p['bk'], p['bh']
                  ) - rLH(p['kLHr'], LHp)
    dLHb_dt = (1 / p['Vb']) * rLH(p['kLHr'], LHp) - p['kLHcl'] * LHb
    dFSHp_dt = sFSH(Aov, f_kisspeptin(Aov, Eb, p['f0'], p['T_fA'], p['n_fA'], p['T_fE'], p['n_fE']),
                    p['bFSHs'], p['T_FSHA'], p['nFSHA'], p['T_FSHF'], p['nFSHF']) - rFSH(p['kFSHr'], FSHp)
    dFSHb_dt = (1 / p['Vb']) * rFSH(p['kFSHr'], FSHp) - p['kFSHcl'] * FSHb
    dEb_dt = (1 / p['Vb']) * rE(p['kEr'], Eov) - p['kEcl'] * Eb
    dAb_dt = (1 / p['Vb']) * rA(p['kAr'], Aov) - p['kAcl'] * Ab

    return [dEov_dt, dAov_dt, dLHp_dt, dLHb_dt, dFSHp_dt, dFSHb_dt, dEb_dt, dAb_dt]

# Initial conditions
y0 = [0.08, 5, 4, 0, 3, 0, 0, 0]

# Time points
t_span = (0, 100)
t_eval = np.linspace(*t_span, 500)

# Solve ODEs
sol = solve_ivp(model_odes, t_span, y0, args=(model.parameters,), t_eval=t_eval)

# Calculate f_ and a_kisspeptin at each time point
f_kisspeptin_values = [model.f_kisspeptin(Ab, Eb) for Ab, Eb in zip(sol.y[7], sol.y[6])]
a_kisspeptin_values = [model.a_kisspeptin(Eb) for Eb in sol.y[6]]

# Plot results
import matplotlib.pyplot as plt

#plt.plot(sol.t, sol.y.T)
#plt.xlabel('Time')
#plt.legend(['Eov', 'Aov', 'LHp', 'LHb', 'FSHp', 'FSHb', 'Eb', 'Ab', 'f_kisspeptin'])
#plt.ylabel('Concentration')
#plt.title('Hormonal Dynamics Over Time')
#plt.show()

# Set of subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

axs[0, 0].plot(sol.t, sol.y[6], label='Eb')
axs[0, 0].set_title('Eb')
axs[0, 0].legend()

axs[0, 1].plot(sol.t, sol.y[7], label='Ab')
axs[0, 1].set_title('Ab')
axs[0, 1].legend()

axs[1, 0].plot(sol.t, sol.y[3], label='LHb')
axs[1, 0].set_title('LHb')
axs[1, 0].legend()

axs[1, 1].plot(sol.t, sol.y[4], label='FSHb')
axs[1, 1].set_title('FSHb')
axs[1, 1].legend()

axs[2, 0].plot(sol.t, a_kisspeptin_values, label='a_kisspeptin')
axs[2, 0].set_title('a_kisspeptin')
axs[2, 0].legend()

axs[2, 1].plot(sol.t, f_kisspeptin_values, label='f_kisspeptin')
axs[2, 1].set_title('f_kisspeptin')
axs[2, 1].legend()

# Adjust the layout
plt.tight_layout()
plt.show()