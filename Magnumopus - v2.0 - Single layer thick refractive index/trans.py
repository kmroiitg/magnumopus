from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal.windows import blackman

def trans(t1,t2,f1,f2,ref,sample,t3,t4):
        
    time_start = t1              #full length is 0.00 to 699.90 ps
    time_end   = t2
    freq1      = f1
    freq2      = f2  

    data_begin  = 6               #beginning data line in picotd/picofd file 
    
    time_ref = []
    voltage_ref = []
    time_sample = []
    voltage_sample = []
    time_ref_full = []
    volt_ref_full = []
    time_sam_full = []
    volt_sam_full = []

    line_number = 1
    for line in ref:
        if line_number < data_begin:
            line_number+=1
        else:
            lines = [i for i in line.split()]
            time_ref_full.append(float(lines[0]))
            volt_ref_full.append(float(lines[1]))
            if float(lines[0]) >= time_start and float(lines[0]) <= time_end:
                time_ref.append(float(lines[0]))
                voltage_ref.append(float(lines[1]))
            else:
                continue
                
    line_number = 1
    for line in sample:
        if line_number < data_begin:
            line_number+=1
        else:
            lines = [i for i in line.split()]
            time_sam_full.append(float(lines[0]))
            volt_sam_full.append(float(lines[1]))
            if float(lines[0]) >= t3 and float(lines[0]) <= t4:
                time_sample.append(float(lines[0]))
                voltage_sample.append(float(lines[1]))
            else:
                continue
            
    #Baseline correction
    base_time_end = 279.0 

    N=0
    summ = 0.0
    for i in volt_ref_full:
        if time_ref_full[N] <= base_time_end:
            summ+=i
            N+=1
        else: break

    offset_ref = summ/N
    
    print('offset ref:',offset_ref)

    N=0
    summ = 0.0
    for i in volt_sam_full:
        if time_sam_full[N] <= base_time_end:
            summ+=i
            N+=1
        else: break

    offset_sample = summ/N
    
    print('offset sam:',offset_sample)
    
    
    index = 0
    for i in voltage_ref:
        voltage_ref[index] = i-offset_ref
        index+=1

    index = 0
    for i in voltage_sample:
        voltage_sample[index] = i-offset_sample
        index+=1
        
    N = len(time_ref) # Number of sample points reference
    N2 = len(time_sample)

    #zero pad
    res = 1e9
    samp_rate = 10e12
    samp_points = int(samp_rate/res)
    N_req = samp_points - N
    N_req_sam = samp_points - N2
    left_pad = int(N_req//2)
    right_pad = int(N_req) - left_pad
    left_pad_sam = int(N_req_sam//2)
    right_pad_sam = int(N_req_sam) - left_pad_sam
    
    voltage_ref_padded = np.pad(voltage_ref,(left_pad,right_pad))
    voltage_sam_padded = np.pad(voltage_sample,(left_pad_sam,right_pad_sam))

    N = len(voltage_ref_padded)
    #N = len(voltage_ref)  #Enable for unpadded calculations
    
    y1 = voltage_ref_padded
    y2 = voltage_sam_padded

    #y1 = voltage_ref  #Enable for unpadded calculations
    #y2 = voltage_sample    #Enable for unpadded calculations
    w = blackman(N2)

    T   = np.mean(np.diff(time_ref_full))   #To find time step
    yf1 = fft(y1)
    yf2 = fft(y2)
    xf = fftfreq(N, T)[:N//2]
    #Finding index of freq1 and freq2
    index_freq1 = np.where(xf >= freq1)[0][0]
    index_freq2 = np.where(xf <= freq2)[0][len(np.where(xf <= freq2)[0])-1]

    #sliced frequency
    xf_sliced = xf[index_freq1:index_freq2]

    #Find temporal position of absolute of maximum in time signals
    var1 = np.abs(voltage_ref)
    var2 = np.abs(voltage_sample)
    index_ref_max = np.where(var1 == max(var1))
    index_sample_max = np.where(var2 == max(var2))
    time0_ref = time_ref[int(index_ref_max[0])]
    time0_sam = time_sample[int(index_sample_max[0])]

    #Time shift between peaks of ref and sample
    amp_ratio = np.max(var2)/np.max(var1)
    time_shift = time0_sam - time0_ref

    #reduced phase
    phi0_ref = 2*np.pi*xf*time0_ref
    phi0_sam = 2*np.pi*xf*time0_sam
    phi0_ref_sliced = phi0_ref[index_freq1:index_freq2]
    phi0_sam_sliced = phi0_sam[index_freq1:index_freq2]
    phi_ref1 = np.angle(yf1[:N//2]*np.exp(-1*complex(0,1)*phi0_ref))
    phi_sam1 = np.angle(yf2[:N//2]*np.exp(-1*complex(0,1)*phi0_sam))

    phi_ref = phi_ref1[index_freq1:index_freq2]
    phi_sam = phi_sam1[index_freq1:index_freq2]
    #unwrapping reduced phases
    dphi0_1 = np.unwrap(phi_sam - phi_ref)

    #checking for global phase offset in the dynamic range of FFT
    slope, intercept, r, p, std_err = stats.linregress(xf_sliced, dphi0_1)

    #removing phase offset
    dphi_0 = dphi0_1 - 2*np.pi*round(intercept/(2*np.pi))

    #final full phase difference
    dphi_1 = dphi_0 - phi0_ref_sliced + phi0_sam_sliced
    
    
    print('ref peak time=',time0_ref)
    print('sam peak time=',time0_sam)
            
    #Calculation of omega and t_no_echo
    ang_freq = 2*np.pi*xf_sliced*1e12
    t = (yf2[index_freq1:index_freq2])/(yf1[index_freq1:index_freq2])
    dphi = np.unwrap(np.angle(t))

    """#Windowing plot
    plt.plot(voltage_ref, label = 'Ref')
    plt.plot(voltage_sample , label = 'Sam')
    plt.plot(w , label = 'Window')
    plt.legend()
    plt.show()"""
    
    plt.figure(0)
    #Time domain plot
    plt.subplot(2,1,1)
    plt.plot(time_ref,voltage_ref)
    plt.plot(time_sample,voltage_sample)
    plt.legend(["Reference", "Sample"], loc = 'upper right',fontsize = 8)
    plt.title("Time domain signal")
    plt.ylabel('Voltage')
    plt.xlabel('Time delay (ps)')
    plt.axhline(y=0,linestyle = '--',color='darkgray',linewidth = 1)

    #Frequency domain plot
    plt.subplot(2,1,2)
    plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yf1[1:N//2]), label = 'Reference')
    plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yf2[1:N//2]), label = 'Sample')
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.title("Frequency domain signal")
    plt.ylabel('Magnitude (arb. units)')
    plt.xlabel('THz')
    plt.tight_layout()
    plt.savefig('tf_fd.png',dpi=300)
    plt.show()

    # plt.figure(1)
    # #Time domain plot Padded
    # plt.plot(voltage_ref_padded)
    # plt.plot(voltage_sam_padded)
    # plt.legend(["Reference", "Sample"], loc = 'upper right',fontsize = 8)
    # plt.title("Time domain signal")
    # plt.ylabel('Voltage')
    # plt.axhline(y=0,linestyle = '--',color='darkgray',linewidth = 1)
    # plt.show()

    class exp_data:
        def __init__(self, t1, t2,v1,v2):
            self.time_ref = t1
            self.time_sam = t2
            self.v_ref    = v1
            self.v_sam    = v2
    data = exp_data(time_ref,time_sample,voltage_ref,voltage_sample)
    
    return t, dphi , ang_freq , data , dphi_1 , time_shift, amp_ratio
