
from T import *
from trans import *
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import datetime
##############################################################################################################

ref = open(r"F:\THz Data\rohith\SSPL\for nelly\td_subs.picotd")
sam = open(r"F:\THz Data\rohith\SSPL\for nelly\td_s2.picotd")

t1 = 296.38
t2 = 315.7          #For reference

t3 = 296.38
t4 = 315.7          #For sample

f1 = 0.2
f2 = 1.5

n1  = complex(1.0,0.0)        #initial guess
n2  = [500,500]
n3  = complex(3.43,0.0)

L = 1000e-9

################################################################################################################
t,dphi,omega,data = trans(t1,t2,f1,f2,ref,sam,t3,t4)


n_r = np.zeros(len(omega),dtype=float)
n_i = np.zeros(len(omega),dtype=float)
t_theo = np.zeros(len(omega),dtype = complex)
dM = np.zeros(len(omega),dtype=float)
dPhi = np.zeros(len(omega),dtype=float)


def min_func(x,L,omega,t,t_theo,dphi):

    t_theo[k] = T(n1,x,n3,omega,L)
    M = np.log(np.abs(t_theo[k]) / np.abs(t[k]))
    Phi = np.unwrap(np.angle(t_theo[:k + 1]))[k] - dphi[k]
    dM[k] = M
    dPhi[k] = Phi
    xi = 1.0
    return np.linalg.norm(dM)+xi*np.linalg.norm(dPhi)
    #return M**2 + 1.0*Phi**2


k=0
for j in omega:
    res = scipy.optimize.minimize(min_func,
                           x0=n2,
                           args=(L,j,t,t_theo,dphi),
                           method = 'Nelder-Mead',
                           options={'xatol': 1e-8, 'disp': True}
                           )

    n2   = [res.x[0],res.x[1]]
    
    n_r[k] = res.x[0]
    n_i[k] = res.x[1]
    k+=1

k=0
for j in omega:
    n2 = [n_r[k],n_i[k]]
    res = scipy.optimize.minimize(min_func,
                           x0=n2,
                           args=(L,j,t,t_theo,dphi),
                           method = 'Nelder-Mead',
                           options={'xatol': 1e-9, 'disp': True}
                           )

    n_r[k] = res.x[0]
    n_i[k] = res.x[1]
    k+=1


freq = omega/(2*np.pi)
freq_THz = freq/1e12

ekk = datetime.datetime.now()
d,m,y,hr,mi = ekk.day,ekk.month,ekk.year,ekk.hour,ekk.minute
day_time = str(d)+'-'+str(m)+'-'+str(y)+' '+str(hr)+':'+str(mi)
mat_info = 'n3='+str(n3)+' L2='+str(L*1e6)+'um  '
font_xaxis = 7.5
font_yaxis = 7.5
font_title = 8
font_legend = 8

plt.subplot(2,2,1)
plt.plot(freq_THz,n_r,label='Real')
plt.plot(freq_THz,n_i,label='Imag')
plt.xlabel('Frequency (THz)',fontsize = font_xaxis)
plt.ylabel('Complex refractive index',fontsize = font_yaxis)
plt.legend(fontsize = font_legend)
plt.axhline(y=0,linestyle = '--',color='darkgray',linewidth = 1)
#plt.savefig('nk.png',dpi=300)
#plt.show()
#plt.close()

plt.subplot(2,2,2)
plt.plot(freq_THz,np.abs(t),label='Exp')
plt.plot(freq_THz,np.abs(t_theo),label='Theory')
plt.xlabel('Frequency (THz)',fontsize = font_xaxis)
plt.ylabel('Mag(t)',fontsize = font_yaxis)
plt.legend(fontsize = font_legend)
#plt.savefig('t_mag.png',dpi=300)
#plt.show()
#plt.close()

plt.subplot(2,2,4)
plt.plot(data.time_ref,data.v_ref,label='Ref')
plt.plot(data.time_sam,data.v_sam,label='Sam')
plt.xlabel('Time (ps) ['+str(np.min(data.time_ref))+'ps - '+str(np.max(data.time_ref))+'ps]' ,fontsize = font_xaxis)
plt.ylabel('Voltage',fontsize = font_yaxis)
plt.legend(fontsize = font_legend)
plt.axhline(y=0,linestyle = '--',color='darkgray',linewidth = 1)

plt.subplot(2,2,3)
plt.plot(freq_THz,np.angle(t),label='Exp')
plt.plot(freq_THz,np.angle(t_theo),label='Theory')
plt.xlabel('Frequency (THz)',fontsize = font_xaxis)
plt.ylabel('Phase(t)',fontsize = font_yaxis)
plt.legend(fontsize = font_legend)
plt.suptitle(mat_info+day_time,fontsize = font_title)
plt.tight_layout()
plt.savefig('',dpi=300)
plt.show()
plt.close()
