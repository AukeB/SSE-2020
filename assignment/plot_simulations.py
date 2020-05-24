# Stellar Structure & Evolution 2020: Practical Assignment.
# Auke Bruinsma (s1594443).

 ### Import packages ###

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import os
from decimal import Decimal
import regimes


 ### Global constants ###

n1 = 76 # Number of .data files which is the same as the number of time instances of the star.
n2 = 125 # For M_1
figsize=(10,7)

 ### Set up paths and directories ###

# General data directory.
#data_dir = '/data2/bruinsma/SSE/data'
data_dir = '/home/auke/Desktop/temp/data'

# Two different paths for the two different simulations.
sim_1 = '/1m_sun/LOGS/'
sim_2 = '/2m_sun/LOGS/'
numpy_arrays = '/numpy_arrays/'

 ### Load all parameters so that you only have to go once through all time instances ###

star_age_1 = np.load(data_dir+numpy_arrays+'star_age_1.npy')
star_age_2 = np.load(data_dir+numpy_arrays+'star_age_2.npy')
num_zones_1 = np.load(data_dir+numpy_arrays+'num_zones_1.npy')
num_zones_2 = np.load(data_dir+numpy_arrays+'num_zones_2.npy')
Teff_1 = np.load(data_dir+numpy_arrays+'Teff_1.npy')
Teff_2 = np.load(data_dir+numpy_arrays+'Teff_2.npy')
photosphere_L_1 = np.load(data_dir+numpy_arrays+'photosphere_L_1.npy')
photosphere_L_2 = np.load(data_dir+numpy_arrays+'photosphere_L_2.npy')
logT_1 = np.load(data_dir+numpy_arrays+'logT_1.npy')
logT_2 = np.load(data_dir+numpy_arrays+'logT_2.npy')
logRho_1 = np.load(data_dir+numpy_arrays+'logRho_1.npy')
logRho_2 = np.load(data_dir+numpy_arrays+'logRho_2.npy')
logR_1 = np.load(data_dir+numpy_arrays+'logR_1.npy')
logR_2 = np.load(data_dir+numpy_arrays+'logR_2.npy')
grada_1 = np.load(data_dir+numpy_arrays+'grada_1.npy')
grada_2 = np.load(data_dir+numpy_arrays+'grada_2.npy')
gradr_1 = np.load(data_dir+numpy_arrays+'gradr_1.npy')
gradr_2 = np.load(data_dir+numpy_arrays+'gradr_2.npy')

#print(logR_1[35].iloc[354]) # Test.
#print(star_age_1)

 ### PLOT: Evolution of the core in the log T_c - log Rho_c plane ###

def make_plot_1():
	def inspect_arrays():
		for i in range(n1):
			print(f'{i} {logT_1[i]} {logRho_1[i]} {star_age_1[i]:.2e}')
		for i in range(n2):
			print(f'{i} {logT_2[i]} {logRho_2[i]} {star_age_2[i]:.2e}')

	#inspect_arrays()

	fig, ax = plt.subplots(1,1,figsize=figsize)

	ax.set_title(r'Evolution of the core in the $\log T_c$ - $\log \rho_c$ plane',fontsize=20)
	ax.set_xlabel(r'$\log T_c$ (K)',fontsize=16)
	ax.set_ylabel(r'$\log \rho_c$ (g cm$^{-3}$)',fontsize=16)

	jet = plt.get_cmap('jet') 
	cNorm_1  = colors.Normalize(vmin = 0, vmax = len(logT_1) - 1)
	cNorm_2 = colors.Normalize(vmin = 0, vmax = len(logT_2) - 1)
	scalarMap_1 = cmx.ScalarMappable(norm = cNorm_1, cmap = jet)
	scalarMap_2 = cmx.ScalarMappable(norm = cNorm_2, cmap = jet)

	for i in range(n1-1):
		colorVal_1 = scalarMap_1.to_rgba(i)

		x_1 = [logT_1[i], logT_1[i+1]]
		y_1 = [logRho_1[i], logRho_1[i+1]]

		if i == 1:
			ax.plot(x_1, y_1, color = colorVal_1,lw=0.7,ls='--',label=r'$1M_{Sun}$')
			ax.scatter(logT_1[i],logRho_1[i],color=colorVal_1,s=1.4)

		ax.plot(x_1, y_1, color = colorVal_1,lw=0.7,ls='--')
		ax.scatter(logT_1[i],logRho_1[i],color=colorVal_1,s=1.4)

	for i in range(n2-1):
		colorVal_2 = scalarMap_2.to_rgba(i)	

		x_2 = [logT_2[i], logT_2[i+1]]
		y_2 = [logRho_2[i], logRho_2[i+1]]

		if i == 1:
			ax.plot(x_2, y_2, color = colorVal_2,lw=0.7,label=r'$2M_{Sun}$')
			ax.scatter(logT_2[i],logRho_2[i],color=colorVal_2,s=1.4)

		ax.plot(x_2, y_2, color = colorVal_2,lw=0.7)
		ax.scatter(logT_2[i],logRho_2[i],color=colorVal_2,s=1.4)

	sm = plt.cm.ScalarMappable(cmap = jet, norm = plt.Normalize(vmin=np.min(star_age_1), vmax=np.max(star_age_1)))
	sm._A = []
	cb = plt.colorbar(sm)
	cb.set_label('Age [Myr]')

	#ax.plot(logT_1,logRho_1,label=r'$1M_{Sun}$',ls='--', lw=0.8, marker='o', ms=4, color='brown')
	#ax.plot(logT_2,logRho_2,label=r'$2M_{Sun}$',ls='--', lw=0.8, marker='o', ms=4, color='k')

	ax.plot(np.log10(regimes.T_rad_ig),regimes.log_rho,':',label='Radiation | Ideal gas',lw=0.7)
	ax.plot(np.log10(regimes.T_ig_NR[0:regimes.b3]),regimes.log_rho[0:regimes.b3],':',label='Ideal gas | NR degenerate',lw=0.7)
	ax.plot(regimes.log_rho[0:regimes.b1],np.log10(regimes.T_NR_ER[0:regimes.b1]),':',label='NR degenerate | ER degenerate',lw=0.7)
	ax.plot(np.log10(regimes.T_ig_ER[regimes.b2:-1]),regimes.log_rho[regimes.b2:-1],':',label='Ideal gas | ER degenerate',lw=0.7)

	ax.text(7.7,-2,s='Radiation',fontsize=12)
	ax.text(6,0,s='Ideal gas',fontsize=12)
	ax.text(5.7,4.3,s='Degenerate NR',fontsize=12)
	ax.text(6,6.4,s='Degenerate ER',fontsize=12)

	# Position solar core.
	ax.scatter(np.log10(15e6), np.log10(150e3), marker = '*', color = 'orange',label='Solar core current position')

	ax.set_xlim(5.2,8.3)
	ax.set_ylim(-4.5,7)

	ax.legend()
	#ax.grid()

	#plt.show()
	plt.savefig('figs/plot1.png')
	plt.close()

 ### PLOT: Hertzsprung-Russel diagram ###

def make_plot_2():
	def inspect_arrays():
		for i in range(n1):
			print(i,Teff_1[i],photosphere_L_1[i])
		for i in range(n2):
			print(i,Teff_2[i],photosphere_L_2[i])

	#inspect_arrays()

	fig, ax = plt.subplots(1,1,figsize=figsize)

	ax.set_title(r'Hertzsprung-Russel Diagram',fontsize=20)
	ax.set_xlabel(r'$T_{eff}$ (K)',fontsize=16)
	ax.set_ylabel(r'$L$ $(L_{Sun}$)',fontsize=16)

	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.gca().invert_xaxis()

	jet = plt.get_cmap('jet') 
	cNorm_1  = colors.Normalize(vmin = 0, vmax = len(Teff_1) - 1)
	cNorm_2 = colors.Normalize(vmin = 0, vmax = len(Teff_2) - 1)
	scalarMap_1 = cmx.ScalarMappable(norm = cNorm_1, cmap = jet)
	scalarMap_2 = cmx.ScalarMappable(norm = cNorm_2, cmap = jet)

	for i in range(n1-1):
		colorVal_1 = scalarMap_1.to_rgba(i)

		x_1 = [Teff_1[i], Teff_1[i+1]]
		y_1 = [photosphere_L_1[i], photosphere_L_1[i+1]]

		if i == 1:
			ax.plot(x_1, y_1, color = colorVal_1,lw=0.7,ls='--',label=r'$1M_{Sun}$')
			ax.scatter(Teff_1[i],photosphere_L_1[i],color=colorVal_1,s=1.4)

		ax.plot(x_1, y_1, color = colorVal_1,lw=0.7,ls='--')
		ax.scatter(Teff_1[i],photosphere_L_1[i],color=colorVal_1,s=1.4)


	for i in range(n2-1):
		colorVal_2 = scalarMap_2.to_rgba(i)	

		x_2 = [Teff_2[i], Teff_2[i+1]]
		y_2 = [photosphere_L_2[i], photosphere_L_2[i+1]]

		if i == 1:
			ax.plot(x_2, y_2, color = colorVal_2,lw=0.7,label=r'$2M_{Sun}$')
			ax.scatter(Teff_2[i],photosphere_L_2[i],color=colorVal_2,s=1.4)

		ax.plot(x_2, y_2, color = colorVal_2,lw=0.7)
		ax.scatter(Teff_2[i],photosphere_L_2[i],color=colorVal_2,s=1.4)


	sm = plt.cm.ScalarMappable(cmap = jet, norm = plt.Normalize(vmin=np.min(star_age_1), vmax=np.max(star_age_1)))
	sm._A = []
	cb = plt.colorbar(sm)
	cb.set_label('Age [Myr]')

	#ax.plot(logT_1,logRho_1,label=r'$1M_{Sun}$',ls='--', lw=0.8, marker='o', ms=4, color='brown')
	#ax.plot(logT_2,logRho_2,label=r'$2M_{Sun}$',ls='--', lw=0.8, marker='o', ms=4, color='k')

	ax.legend()
	#ax.grid()

	#plt.show()
	plt.savefig('figs/plot2.png')
	plt.close()

 ### Convection ###

def make_plot_3_and_4():
	for i in range(n2):
		print(i)
		fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))

		ax1.set_title(r'Adiabatic gradient',fontsize=18)
		ax1.set_xlabel(r'$R (R_{Sun})$',fontsize=16)
		ax1.set_ylabel(r'$\nabla_{ad}$',fontsize=16)

		ax2.set_title(r'Radiative gradient',fontsize=18)
		ax2.set_xlabel(r'$R (R_{Sun})$',fontsize=16)
		ax2.set_ylabel(r'$\nabla_{rad}$',fontsize=16)

		ax1.plot(logR_1[i],grada_1[i],color='brown',lw=1,ls='--',label=r'$1M_{Sun}$')
		ax1.plot(logR_2[i],grada_2[i],c='k',lw=1,ls='-',label=r'$2M_{Sun}$')

		ax2.plot(logR_1[i],gradr_1[i],color='brown',lw=1,ls='--',label=r'$1M_{Sun}$')
		ax2.plot(logR_2[i],gradr_2[i],c='k',lw=1,ls='-',label=r'$2M_{Sun}$')

		ax1.legend()
		ax2.legend()

		plt.savefig(f'figs/convection_plots/plot_convection_profile_{i}.png')
		plt.close()

def main():
	make_plot_1()
	make_plot_2()
	make_plot_3_and_4()

main()