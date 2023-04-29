import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

def Vpot(x):        ###defining the potential(1/2 k X^2)
            return x**2

a = -15. ###initial value of x
b = 15.  ### final value of x
N = 500     ### no. of points b/w initial and final points
x = np.linspace(a,b,N) ### x range
h = x[1]-x[0]   ### step size

p2 = np.zeros((N)**2).reshape(N,N)        ### momentum-squared operator matrix
for i in range(N):
    for j in range(N):
        if i==j:
            p2[i,j]= -2
        elif np.abs(i-j)==1:
            p2[i,j]=1
        else:
            p2[i,j]=0

V = np.zeros((N)**2).reshape(N,N)     ### Potential matrix
for i in range(N):
    for j in range(N-2):
        if i==j:
            V[i,j]= Vpot(x[i+1])
        else:
            V[i,j]=0

H = -p2/(2*h**2) + V/2      ### Hamiltonian matrix

val,vec=np.linalg.eig(H)        ### eigenvalue and eigenvectors of matrix H
z = np.argsort(val)             ### sorting eigenvalues in ascending order
energies=(val[z]/val[z][0])

sigma= 0.7     #### std dev of gaussian
def Gaussian(s):
	return np.exp(-0.5*((s+5.)/sigma)**2)/(sigma*np.sqrt(2*np.pi))		### Gaussian normalized having peak at x=-5
###total wave fn
c=[]    #### projection of wave fn n eigen basis
ps=np.zeros(len(vec[:,0]),dtype=complex)
for i in range(len(vec[:,0])):
    c.append((np.dot(Gaussian(x),vec[:,z[i]])))#### vallue of coefficient ci's
    ps += c[i]*vec[:,z[i]]  ####expressing gaussian as linear sum of eigenbasis
def Psi(t): #### Time evolving wave fn
    w=complex(0,1)
    return np.dot((expm(-w*H*t)),ps)    ### time evolution operator operating on the wavefn
#### setting up axis and axis info on graph
fig, ax=plt.subplots(figsize=(8,6),dpi=300)
ax.set(xlim=(-10.,10.), ylim=(0.,.4))
lines = plt.plot([])
line = lines[0]
plt.xlabel('x', size=14)
plt.ylabel('$\psi^2$(x)',size=14)
plt.title("Time evolving gaussian in Harmonic Oscillator\n std. dev<1")

### animating psi**2 wrt time
def animate(frame):
    line.set_data((x,abs(Psi(frame/100))**2))
anim=FuncAnimation(fig, animate, frames=300, interval= 50)
plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'#### calling the ffmpeg software to make the animation
FFwriter=animation.FFMpegWriter(fps=50)
anim.save('HO2.mp4', writer=FFwriter)### saving animated file


