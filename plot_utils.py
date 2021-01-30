import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess as sp
import os, shutil

def get_linear_system(file):
	data = pd.read_csv(file)
	x = np.sin(data['latitude'].to_numpy() * np.pi / 180) ** 2
	y = data['ratio'].to_numpy()
	w = data['weights'].to_numpy()
	locs = data['location']
	return x, y, w, locs

def labels_2d(ax, legend=False):
	if legend:
		ax.legend()
	ax.set_xlim(-.1, 1)
	ax.set_yticks(range(25500, 26000, 100))
	ax.set_yticklabels([f'{l/1000}k' for l in range(25500, 26000, 100)])
	ax.set_xlabel('$sin^2{\lambda}$ (x)');
	ax.set_ylabel('$\Delta s / \Delta \lambda$ (y)');

def get_filtered_linear_system(file):
	x, y, w, locs = get_linear_system(file)
	filt = locs.isin(('Peru', 'Pennsylvania', 'Lapland'))
	n, k = np.sum(filt), 2
	
	x_filt = np.stack((np.ones(n), x[filt])).transpose() # Column of 1's needed for constant terms
	y_filt = y[filt]

	# Some operations to exagerrate difference between y and its projection
	y_filt[1] = y_filt[1] / 2
	y_filt[2] = y_filt[2] * 2
	w_filt = np.diag(w[filt])
	locs_filtered = locs[filt]

	# Let's plot unit vectors for a cleaner image
	x_filt[:,0] = x_filt[:,0] / np.linalg.norm(x_filt[:,0])
	x_filt[:,1] = x_filt[:,1] / np.linalg.norm(x_filt[:,1])
	y_filt = y_filt / np.linalg.norm(y_filt)

	return x_filt, y_filt, w_filt, locs_filtered

def labels_3d(ax):
	ax.plot(ax.get_xlim(), [0,0], [0,0], color = 'black')
	ax.plot([0, 0], ax.get_ylim(), [0, 0], color = 'black')
	ax.plot([0, 0], [0,0], ax.get_zlim(), color = 'black')

	ax.set_xlim(-.2, 1), ax.set_ylim(-.2, 1)
	ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
	ax.grid(False)

def save_animation(ax):
	os.mkdir('projection_views')
	for i, angle in enumerate(range(0, 360, 3)):
		# Consider changing elavation too
		ax.view_init(elev=10, azim=angle)
		plt.savefig(f'projection_views/{i:03d}.png', dpi=150)

	sp.check_output(['ffmpeg', '-y', '-i', 'projection_views/%03d.png', '-vf', 'palettegen', '-y', 'projection_views/palette.png'])
	sp.check_output(['ffmpeg', '-y', '-i', 'projection_views/%03d.png', '-i', 'projection_views/palette.png', '-lavfi', 'paletteuse', 'projection_views/out.gif'])
	sp.check_output(['convert', 'projection_views/out.gif', '-trim', '-layers', 'trim-bounds', 'images/projection.gif'])
	shutil.rmtree('projection_views')