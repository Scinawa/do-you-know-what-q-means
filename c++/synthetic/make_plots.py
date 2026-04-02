import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots

plt.style.use('science')
plt.rcParams["mathtext.fontset"] = "cm"

n = []
eps_n = []
d = []
eps_d = []
k = []
eps_k = []
n_sampling = []
time_sampling = []

with open('data_n_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line: 
			if float((line.strip('\n').split(" "))[0]) not in n:
				n.append( float((line.strip('\n').split(" "))[0]) )
			if float((line.strip('\n').split(" "))[1]) not in eps_n:
	    			eps_n.append( float((line.strip('\n').split(" "))[1]) )


with open('data_d_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line: 
			if float((line.strip('\n').split(" "))[0]) not in d:
				d.append( float((line.strip('\n').split(" "))[0]) )
			if float((line.strip('\n').split(" "))[1]) not in eps_d:
	    			eps_d.append( float((line.strip('\n').split(" "))[1]) )


with open('data_k_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line: 
			if float((line.strip('\n').split(" "))[0]) not in k:
				k.append( float((line.strip('\n').split(" "))[0]) )
			if float((line.strip('\n').split(" "))[1]) not in eps_k:
	    			eps_k.append( float((line.strip('\n').split(" "))[1]) )
	 
	    			
with open('data_sampling_time_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line: 
			n_sampling.append( float((line.strip('\n').split(" "))[0]) )
			time_sampling.append( float((line.strip('\n').split(" "))[1]) )
    			


data_n = {
	'time': [[] for i in range(len(eps_n))],
	'time_preprocessing': [[] for i in range(len(eps_n)-1)],
	'iterations': [[] for i in range(len(eps_n))],
	'RSS': [[] for i in range(len(eps_n))],
	'ARI': [[] for i in range(len(eps_n))],
	'NMI': [[] for i in range(len(eps_n))],
}

data_d = {
	'time': [[] for i in range(len(eps_d))],
	'iterations': [[] for i in range(len(eps_d))],
	'RSS': [[] for i in range(len(eps_d))],
	'ARI': [[] for i in range(len(eps_d))],
	'NMI': [[] for i in range(len(eps_d))],
}

data_k = {
	'time': [[] for i in range(len(eps_k))],
	'iterations': [[] for i in range(len(eps_k))],
	'RSS': [[] for i in range(len(eps_k))],
	'ARI': [[] for i in range(len(eps_k))],
	'NMI': [[] for i in range(len(eps_k))],
}

with open('data_n_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line:
			n_value = float((line.strip('\n').split(" "))[0])
			eps_value = float((line.strip('\n').split(" "))[1])
			
			i = 0
			while eps_n[i] != eps_value:
				i += 1 
			
			time = float((line.strip('\n').split(" "))[2])
			data_n['time'][i].append( time / 1000 )
			
			iterations = float((line.strip('\n').split(" "))[3])
			data_n['iterations'][i].append( iterations )
			
			if eps_value != 0.0:
				RSS = float((line.strip('\n').split(" "))[4])
				ARI = float((line.strip('\n').split(" "))[5])
				NMI = float((line.strip('\n').split(" "))[6])
				data_n['RSS'][i].append( RSS )
				data_n['ARI'][i].append( ARI )
				data_n['NMI'][i].append( NMI )
				

with open('data_n_preprocessing_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line:
			n_value = float((line.strip('\n').split(" "))[0])
			eps_value = float((line.strip('\n').split(" "))[1])
			
			i = 0
			while eps_n[i] != eps_value:
				i += 1
				
			time = float((line.strip('\n').split(" "))[2])
			data_n['time_preprocessing'][i-1].append( time / 1000 )
	 
	
with open('data_d_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line:
			d_value = float((line.strip('\n').split(" "))[0])
			eps_value = float((line.strip('\n').split(" "))[1])
			
			i = 0
			while eps_d[i] != eps_value:
				i += 1 
			
			time = float((line.strip('\n').split(" "))[2])
			data_d['time'][i].append( time / 1000 )
			
			iterations = float((line.strip('\n').split(" "))[3])
			data_d['iterations'][i].append( iterations )
			
			if eps_value != 0.0:
				RSS = float((line.strip('\n').split(" "))[4])
				ARI = float((line.strip('\n').split(" "))[5])
				NMI = float((line.strip('\n').split(" "))[6])
				data_d['RSS'][i].append( RSS )
				data_d['ARI'][i].append( ARI )
				data_d['NMI'][i].append( NMI )
	 
	 	
with open('data_k_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line:
			k_value = float((line.strip('\n').split(" "))[0])
			eps_value = float((line.strip('\n').split(" "))[1])
			
			i = 0
			while eps_k[i] != eps_value:
				i += 1 
			
			time = float((line.strip('\n').split(" "))[2])
			data_k['time'][i].append( time / 1000 )
			
			iterations = float((line.strip('\n').split(" "))[3])
			data_k['iterations'][i].append( iterations )
			
			if eps_value != 0.0:
				RSS = float((line.strip('\n').split(" "))[4])
				ARI = float((line.strip('\n').split(" "))[5])
				NMI = float((line.strip('\n').split(" "))[6])
				data_k['RSS'][i].append( RSS )
				data_k['ARI'][i].append( ARI )
				data_k['NMI'][i].append( NMI )
				
				
###############################################################################

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['time'][0], label=r'$\varepsilon$ = '+str(eps_n[0])+' (Lloyd)', color='k', marker='o', linewidth=2.0, markersize=9)

ax.plot(n, data_n['time'][1], label=r'$\varepsilon$ = '+str(eps_n[1])+' (no prepro)', color='blue', marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][2], label=r'$\varepsilon$ = '+str(eps_n[2])+' (no prepro)', color='green', marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][3], label=r'$\varepsilon$ = '+str(eps_n[3])+' (no prepro)', color='orange', marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][4], label=r'$\varepsilon$ = '+str(eps_n[4])+' (no prepro)', color='red', marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['time'][5], label=r'$\varepsilon$ = '+str(eps_n[5])+' (no prepro)', color='purple', marker='*', linewidth=2.0, markersize=11)

ax.plot(n, data_n['time_preprocessing'][0], label=r'$\varepsilon$ = '+str(eps_n[1])+' (prepro)', color='blue', marker='v', linestyle='dotted', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time_preprocessing'][1], label=r'$\varepsilon$ = '+str(eps_n[2])+' (prepro)', color='green', marker='^', linestyle='dotted', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time_preprocessing'][2], label=r'$\varepsilon$ = '+str(eps_n[3])+' (prepro)', color='orange', marker='s', linestyle='dotted', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time_preprocessing'][3], label=r'$\varepsilon$ = '+str(eps_n[4])+' (prepro)', color='red', marker='x', linestyle='dotted', linewidth=2.0, markersize=11)
ax.plot(n, data_n['time_preprocessing'][4], label=r'$\varepsilon$ = '+str(eps_n[5])+' (prepro)', color='purple', marker='*', linestyle='dotted', linewidth=2.0, markersize=11)

ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Runtime (s)', fontsize=20)
plt.xlim([100000,500000])
legend = ax.legend(loc='upper left', prop={'size': 10}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_time.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('time.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
#plt.plot(N, data['time'][0], label='Standard')
ax.plot(n, data_n['time'][0], label=r'$\varepsilon$ = '+str(eps_n[0])+' (Lloyd)', color='k', marker='o', linewidth=2.0, markersize=9)

ax.plot(n, data_n['time'][1], label=r'$\varepsilon$ = '+str(eps_n[1])+' (no prepro)', color='blue', marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][2], label=r'$\varepsilon$ = '+str(eps_n[2])+' (no prepro)', color='green', marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][3], label=r'$\varepsilon$ = '+str(eps_n[3])+' (no prepro)', color='orange', marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][4], label=r'$\varepsilon$ = '+str(eps_n[4])+' (no prepro)', color='red', marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['time'][5], label=r'$\varepsilon$ = '+str(eps_n[5])+' (no prepro)', color='purple', marker='*', linewidth=2.0, markersize=11)

ax.plot(n, data_n['time_preprocessing'][0], label=r'$\varepsilon$ = '+str(eps_n[1])+' (prepro)', color='blue', marker='v', linestyle='dotted', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time_preprocessing'][1], label=r'$\varepsilon$ = '+str(eps_n[2])+' (prepro)', color='green', marker='^', linestyle='dotted', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time_preprocessing'][2], label=r'$\varepsilon$ = '+str(eps_n[3])+' (prepro)', color='orange', marker='s', linestyle='dotted', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time_preprocessing'][3], label=r'$\varepsilon$ = '+str(eps_n[4])+' (prepro)', color='red', marker='x', linestyle='dotted', linewidth=2.0, markersize=11)
ax.plot(n, data_n['time_preprocessing'][4], label=r'$\varepsilon$ = '+str(eps_n[5])+' (prepro)', color='purple', marker='*', linestyle='dotted', linewidth=2.0, markersize=11)

plt.yscale("log")
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Runtime (s)', fontsize=20)
plt.xlim([100000,500000])
legend = ax.legend(loc='lower right', prop={'size': 10}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_time_log.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('time_log.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['iterations'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(n, data_n['iterations'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['iterations'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['iterations'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['iterations'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['iterations'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Number of iterations', fontsize=20)
plt.xlim([100000,500000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_iterations.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('RSS.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['RSS'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['RSS'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['RSS'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['RSS'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['RSS'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Relative Residual Sum of Squares ($\Delta$RSS) (\%)', fontsize=20)
plt.xlim([100000,500000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_RSS.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('RSS.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['ARI'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['ARI'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['ARI'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['ARI'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['ARI'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=20)
plt.xlim([100000,500000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_ARI.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['NMI'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['NMI'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['NMI'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['NMI'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['NMI'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Normalized Mutual Information (NMI)', fontsize=20)
plt.xlim([100000,500000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_NMI.pdf', format='pdf', bbox_inches='tight')
plt.close()

###############################################################################

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(d, data_d['time'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(d, data_d['time'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(d, data_d['time'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(d, data_d['time'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(d, data_d['time'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(d, data_d['time'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dimension $d$', fontsize=20)
ax.set_ylabel('Runtime (s)', fontsize=20)
plt.xlim([10,60])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/d_time.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('time.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
#plt.plot(N, data['time'][0], label='Standard')
ax.plot(d, data_d['time'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(d, data_d['time'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(d, data_d['time'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(d, data_d['time'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(d, data_d['time'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(d, data_d['time'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
plt.yscale("log")
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dimension $d$', fontsize=20)
ax.set_ylabel('Runtime (s)', fontsize=20)
plt.xlim([10,60])
legend = ax.legend(loc='lower right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/d_time_log.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('time_log.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(d, data_d['iterations'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(d, data_d['iterations'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(d, data_d['iterations'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(d, data_d['iterations'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(d, data_d['iterations'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(d, data_d['iterations'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dimension $d$', fontsize=20)
ax.set_ylabel('Number of iterations', fontsize=20)
plt.xlim([10,60])
legend = ax.legend(loc='upper right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/d_iterations.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('RSS.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(d, data_d['RSS'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(d, data_d['RSS'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(d, data_d['RSS'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(d, data_d['RSS'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(d, data_d['RSS'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dimension $d$', fontsize=20)
ax.set_ylabel('Relative Residual Sum of Squares ($\Delta$RSS) (\%)', fontsize=20)
plt.xlim([10,60])
legend = ax.legend(loc='upper right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/d_RSS.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('RSS.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(d, data_d['ARI'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(d, data_d['ARI'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(d, data_d['ARI'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(d, data_d['ARI'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(d, data_d['ARI'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dimension $d$', fontsize=20)
ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=20)
plt.xlim([10,60])
legend = ax.legend(loc='lower right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/d_ARI.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(d, data_d['NMI'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(d, data_d['NMI'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(d, data_d['NMI'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(d, data_d['NMI'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(d, data_d['NMI'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Dimension $d$', fontsize=20)
ax.set_ylabel('Normalised Mutual Information (NMI)', fontsize=20)
plt.xlim([10,60])
legend = ax.legend(loc='lower right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/d_NMI.pdf', format='pdf', bbox_inches='tight')
plt.close()

###############################################################################

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(k, data_k['time'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(k, data_k['time'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(k, data_k['time'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(k, data_k['time'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(k, data_k['time'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(k, data_k['time'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Number of centroids $k$', fontsize=20)
ax.set_ylabel('Runtime (s)', fontsize=20)
plt.xlim([2,12])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/k_time.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('time.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
#plt.plot(N, data['time'][0], label='Standard')
ax.plot(k, data_k['time'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(k, data_k['time'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(k, data_k['time'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(k, data_k['time'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(k, data_k['time'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(k, data_k['time'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
plt.yscale("log")
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Number of centroids $k$', fontsize=20)
ax.set_ylabel('Runtime (s)', fontsize=20)
plt.xlim([2,12])
legend = ax.legend(loc='lower right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/k_time_log.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('time_log.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(k, data_k['iterations'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(k, data_k['iterations'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(k, data_k['iterations'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(k, data_k['iterations'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(k, data_k['iterations'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(k, data_k['iterations'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Number of centroids $k$', fontsize=20)
ax.set_ylabel('Number of iterations', fontsize=20)
plt.xlim([2,12])
legend = ax.legend(loc='lower right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/k_iterations.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('RSS.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(k, data_k['RSS'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(k, data_k['RSS'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(k, data_k['RSS'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(k, data_k['RSS'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(k, data_k['RSS'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Number of centroids $k$', fontsize=20)
ax.set_ylabel('Relative Residual Sum of Squares ($\Delta$RSS) (\%)', fontsize=20)
plt.xlim([2,12])
legend = ax.legend(loc='upper right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/k_RSS.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('RSS.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(k, data_k['ARI'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(k, data_k['ARI'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(k, data_k['ARI'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(k, data_k['ARI'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(k, data_k['ARI'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Number of centroids $k$', fontsize=20)
ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=20)
plt.xlim([2,12])
legend = ax.legend(loc='upper right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/k_ARI.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(k, data_k['NMI'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(k, data_k['NMI'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(k, data_k['NMI'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(k, data_k['NMI'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(k, data_k['NMI'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
ax.set_xlabel('Number of centroids $k$', fontsize=20)
ax.set_ylabel('Normalised Mutual Information (NMI)', fontsize=20)
plt.xlim([2,12])
legend = ax.legend(loc='upper right', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/k_NMI.pdf', format='pdf', bbox_inches='tight')
plt.close()

###############################################################################

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n_sampling, time_sampling, marker='o', linewidth=2.0, markersize=9)
ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2, labelsize=17)
ax.tick_params(axis='both', which='minor', direction='inout', length=4, width=1)
plt.xscale("log")
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Runtime (ms)', fontsize=20)
#plt.xlim([2,12])
#legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/sampling_time.pdf', format='pdf', bbox_inches='tight')
plt.close()


