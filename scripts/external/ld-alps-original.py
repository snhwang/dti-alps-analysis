#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:28:27 2023

@author: fordb
new_DWI-ALPS - FORD streamlined
modified for processing data originally processed by:
    Paper: Liu, X, Barisano, G, et al., Cross-Vendor Test-Retest Validation of Diffusion Tensor Image Analysis along the Perivascular Space (DTI-ALPS) for Evaluating Glymphatic System Function, Aging and Disease (2023). DOI: https://doi.org/10.14336/AD.2023.0321-2
    Link to this repository: https://github.com/gbarisano/alps/
"""

#TODO the cubic 2d interpolator does not perform particularly well with data
#that has the same vectors collected multiple times. For isntance in the nret 
#dataset, the same set of vectors is colelcted 3x. the interpolator works well
#before eddy rotates the bvecs, because each iteration remains at the exact
#same point. However, after eddy rotates them, they are ever so slightly misaligned with
#eachother, and this seems to produce some very high-spatial-frequency edging
# I am considering realigning the bvecs, or reaveraging them with a small kernel
#I think this interpolator dealing with the very high spatial frequency data
#is what is producing negative ADCs/.
#2023-12-27 For now, I will use the non-eddy-rotated (i.e. original) bvecs. Because I primarily
#use them for the interpolator, this should improve the interpolated accuracy considerably. 

#TODO this paper loks like their ROIs are more centrally located
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9441421/

#also, I could do the mean V1 vector for each instead of doing the voxelwise
#vectors, either way probably need a way to depict the set of vectors used
#for QC purposes

baseDir="/media/fordb/mneme/2024_alps_adni/ADNI_niis/"

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.cluster import DBSCAN
from scipy.interpolate import CloughTocher2DInterpolator, griddata

#processing booleans

#This determines if voxels within each ROI are dropped if their V1 vectors
#are not consistent with others in the ROI
voxVecClean = True

#printing info and plots
diagnostics = False
diagnosticsHist = False
diagnositcsVecs = False

def load_nifti(file_path):
	"""Load a NIfTI file."""
	return nib.load(file_path).get_fdata()

def normalize_vector(v):
		"""Normalize a vector to unit length."""
		norm = np.linalg.norm(v)
		if norm == 0: 
		   return v
		return v / norm
def greatCircleDistance(v1, v2, polarityInvariant=True, returnDegrees=False):
	"""Calculate the great circle distance between two vectors on a unit sphere."""
	# Normalize the vectors
	v1_norm = normalize_vector(v1)
	v2_norm = normalize_vector(v2)
	angleRads = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
	if polarityInvariant:
		angleRads = np.arccos(np.clip(np.abs(np.dot(v1_norm, v2_norm)), 0.0, 1.0))
	if returnDegrees:
		return np.rad2deg(angleRads)
	else:
		return angleRads

def greatCircleDistanceMatrix(set1, set2=None):
	if set2 == None:
		set2 = set1
	"""Create a distance matrix between two sets of vectors."""
	matrix = np.zeros((len(set1), len(set2)))

	for i, vec1 in enumerate(set1):
		for j, vec2 in enumerate(set2):
			matrix[i, j] = greatCircleDistance(vec1, vec2)
	return matrix

def plot_quiver4(vecs, vals, title, dbscanLabels):
	"""Plot a quiver plot using bvecs and ADC values."""
	X, Y, Z = vecs
	x = X[dbscanLabels == 0] * vals
	y = Y[dbscanLabels == 0] * vals
	z = Z[dbscanLabels == 0] * vals
	xn = X[dbscanLabels != 0] * vals
	yn = Y[dbscanLabels != 0] * vals
	zn = Z[dbscanLabels != 0] * vals
	mm = np.max(np.abs([x,y,z]))
	oMin = -mm
	oMax = mm
	fig = plt.figure(figsize=(12, 12))
	# Original plot
	
	ax1 = fig.add_subplot(221, projection='3d')
	ax1.quiver(0, 0, 0, xn, yn, zn, arrow_length_ratio=0, color='gray') 
	ax1.quiver(0, 0, 0, -xn, -yn, -zn, arrow_length_ratio=0, color='gray') 
	ax1.quiver(0, 0, 0, x, y, z, arrow_length_ratio=0) 
	ax1.quiver(0, 0, 0, -x, -y, -z, arrow_length_ratio=0) 
	ax1.set_title("Proj +madefullshell")
	ax1.set_xlim([oMin, oMax])
	ax1.set_ylim([oMin, oMax])
	ax1.set_zlim([oMin, oMax])
	#set_limits(ax1, x, y, z)
	# View from X-axis
	ax2 = fig.add_subplot(222)
	orig = np.zeros(len(x))
	orign = np.zeros(len(xn))
	ax2.axhline(0, ls=':', c='gray')
	ax2.axvline(0, ls=':', c='gray')
	ax2.quiver(orign, orign, yn, zn, angles='xy', scale_units='xy', scale=1, color='gold') 
	ax2.quiver(orign, orign, -yn, -zn, angles='xy', scale_units='xy', scale=1, color='gold') 
	ax2.quiver(orig, orig, y, z, angles='xy', scale_units='xy', scale=1, color='green') 
	ax2.quiver(orig, orig, -y, -z, angles='xy', scale_units='xy', scale=1, color='green') 
	ax2.set_title("View from X-axis +madefullshell")
	ax2.set_xlim([oMin, oMax])
	ax2.set_ylim([oMin, oMax])
	# View from Y-axis
	ax3 = fig.add_subplot(223)
	ax3.axhline(0, ls=':', c='gray')
	ax3.axvline(0, ls=':', c='gray')
	ax3.quiver(orign, orign, xn, zn, angles='xy', scale_units='xy', scale=1, color='gold')
	ax3.quiver(orign, orign, -xn, -zn, angles='xy', scale_units='xy', scale=1, color='gold')
	ax3.quiver(orig, orig, x, z, angles='xy', scale_units='xy', scale=1, color='green')
	ax3.quiver(orig, orig, -x, -z, angles='xy', scale_units='xy', scale=1, color='green')
	ax3.set_title("View from Y-axis +madefullshell")
	ax3.set_xlim([oMin, oMax])
	ax3.set_ylim([oMin, oMax])

	# View from Z-axis
	ax4 = fig.add_subplot(224)
	ax4.axhline(0, ls=':', c='gray')
	ax4.axvline(0, ls=':', c='gray')
	ax4.quiver(orign, orign, xn, yn, angles='xy', scale_units='xy', scale=1, color='gold')
	ax4.quiver(orign, orign, -xn, -yn, angles='xy', scale_units='xy', scale=1, color='gold')
	ax4.quiver(orig, orig, x, y, angles='xy', scale_units='xy', scale=1, color='green')
	ax4.quiver(orig, orig, -x, -y, angles='xy', scale_units='xy', scale=1, color='green')
	ax4.set_title("View from Z-axis +madefullshell")
	ax4.set_xlim([oMin, oMax])
	ax4.set_ylim([oMin, oMax])
	#plt.tight_layout()
	plt.suptitle(title)
	plt.show()

def orthogonal_component(vector, reference):
	"""Return the component of 'vector' that is orthogonal to 'reference'."""
	v = normalize_vector(vector)
	r = normalize_vector(reference)
	print(v - np.dot(v, r), v - np.dot(v, r) * r)
	return v - np.dot(v, r) * r

def find_tangential_vectors(point):
	"""Find two orthogonal vectors tangent to the sphere at the given point."""
	# Normalize the point vector
	n = normalize_vector(point)

	# Create an arbitrary vector different from n
	if n[0] != 1 or n[1] != 0 or n[2] != 0:
		v = np.array([1, 0, 0])
	else:
		v = np.array([0, 1, 0])

	# Compute the first tangential vector
	u = np.cross(n, v)
	u = normalize_vector(u)

	# Compute the second tangential vector
	w = np.cross(n, u)
	w = normalize_vector(w)

	return n, u, w
def orthographic_projection (inVector, originVector):
	#reflects close hemi values to far hemi
	n, u, w = find_tangential_vectors(originVector)
	vec = normalize_vector(inVector)
	if np.dot(vec, n) > 0:
		#flip to other hemi
		vec = vec * -1
	x, y = np.dot(vec, u), np.dot(vec, w)
	return x, y

def batch_orthographic_projection(vectors, originVector):
	#is this actually equatorial or is it orthographic?
	#thinking from this https://upload.wikimedia.org/wikipedia/commons/b/ba/Comparison_azimuthal_projections.svg
	#I think its orthographic
	##TODO - I could test different projection - interpolations combinations?
	projected_data = []
	for vec in vectors:
		projected_data.append(orthographic_projection(vec, originVector))
	return projected_data

def plot_mesh2(coordinates, values):
	# Convert to x, y, and z lists
	x = [coord[0] for coord in coordinates]
	y = [coord[1] for coord in coordinates]
	z = values

	# Create a grid to interpolate data for a smooth surface
	xi = np.linspace(min(x), max(x), 50)
	yi = np.linspace(min(y), max(y), 50)
	X, Y = np.meshgrid(xi, yi)
	Z = griddata((x, y), z, (X, Y), method='cubic') #method{‘linear’, ‘nearest’, ‘cubic’},
	#Zmax = np.nanmax(Z)
	# Create plot
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)
	#surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
	surf = ax.contourf(X, Y, Z, cmap='viridis', extend='both')
	#surf = ax.contour(X, Y, Z, cmap='gray', edgecolor='none', extend='both')
	#surf = ax.imshow(Z, cmap='viridis')
	# Add color bar which maps values to colors
	plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Value')

	# Labeling the plot
	ax.set_xlabel('U Coordinate (arbitrary orthogonal axes)')
	ax.set_ylabel('W Coordinate (arbitrary orthogonal axes)')
	ax.set_title('Interpolated Mesh of projected ADC values')
	ax.set_xlim((-1,1))
	ax.set_ylim((-1,1))
	# put a dot on the maxima?
	#indices = np.where(Z == Zmax)
	#ax.scatter(indices[1], indices[0], color='red',s=.25)
	#plot the actual measured points
	for coordinate in coordinates:
		ax.scatter(coordinate[0],coordinate[1], color='black',s=1)
	plt.show()



#%% load data

roiLabels = ["R_Association", "R_Projection", "L_Association", "L_Projection"]
#ROI 1, 2, 3, 4 are
#1 = r_association, i.e. Superior_longitudinal_fasciculus
#2 = r_projection, i.e. superior corona radiata
#3 = l_association, i.e. Superior_longitudinal_fasciculus
#4 = l_projection, i.e. superior corona radiata
def adcCalc(x, y, z, diffusionArray, bval, b0Mean):
	adc = np.multiply((-1/bval), np.log(np.divide(diffusionArray[x, y, z], b0Mean[x, y, z])))
	return adc


def alpsCalc(prefix, sub):
	print("- Loading data...")
	eddy_file = baseDir+sub+'/eddy_corrected_data.nii.gz'
	eddy_data = load_nifti(eddy_file)
	V1_file = baseDir+sub+'/dti_V1.nii.gz'
	V1_data = load_nifti(V1_file)
	alpsRois_file = baseDir+sub+'/nativeALPSrois.nii.gz'
	alpsRois_data = load_nifti(alpsRois_file)
	
	#eddy_bvecs_file = '/media/fordb/Scratch/analysis_DWI/_niis_nret/'+prefix+sub+'_dwi-eddy.eddy_rotated_bvecs'
	#bvecs_file = '/media/fordb/Scratch/analysis_DWI/_niis/'+prefix+sub+'_dwi.bvec'
	bvecs_file = baseDir+sub+'/eddy_corrected_data.eddy_rotated_bvecs'
	bvals_file = baseDir+sub+'/bval1'
	bvals = np.loadtxt(bvals_file)
	bvecs = np.loadtxt(bvecs_file)
	bvecs_no0 = bvecs[:,bvals != 0]
	#adc formula:
	#ADC = (-1/b) * nb.ln(sDWI / S0) 
	#need to generate an s0 average image from each bval == 0
	b0Mean = np.mean(eddy_data[:,:,:,bvals == 0], axis=-1)
	
	print("- Computing ADC...")
	global adcData 
	adcData = np.empty((b0Mean.shape[0],b0Mean.shape[1],b0Mean.shape[2], np.sum(bvals != 0)))
	
	
	
	#OLD WAY
	#c = 0
	#for i in range(len(bvals)):
	#	if bvals[i] != 0:
	#		adcData[:,:,:,c] = np.multiply((-1/bvals[i]), np.log(np.divide(eddy_data[:,:,:,i], b0Mean)))
	#		c += 1
	#
	#new way
	ROIflat = alpsRois_data > 0
	for index, _ in np.ndenumerate(eddy_data[:,:,:,0]):
		if ROIflat[index]:
			c = 0
			for i in range(len(bvals)):
				if bvals[i] != 0:
					adcData[*index, c] = adcCalc(*index,eddy_data[:,:,:,i], bvals[i],b0Mean)
					c += 1
	#End new way   (161, 122, 34)
	
	if diagnostics:
		plt.hist(greatCircleDistanceMatrix(np.array(bvecs_no0).T).flatten(), bins=50)
		plt.show()
	
	#get ROI indices
	alps_indices = []
	alps_indices.append(np.where(alpsRois_data == 1.))
	alps_indices.append(np.where(alpsRois_data == 2.))
	alps_indices.append(np.where(alpsRois_data == 3.))
	alps_indices.append(np.where(alpsRois_data == 4.))

	
	alps_v1s = []
	for roiNo in range(len(alps_indices)):
		#for each roi, pull v1 from the coordinates specified
		roiV1s = []
		for coordNo in range(len(alps_indices[roiNo][0])):
			v1vec = V1_data[alps_indices[roiNo][0][coordNo], alps_indices[roiNo][1][coordNo], alps_indices[roiNo][2][coordNo], :]
			roiV1s.append(v1vec)
		alps_v1s.append(roiV1s)
	
	
	
	alpsIndicesClean = []
	alpsMedianV1Clean = []
	if voxVecClean:
		print("- Cleaning ROIs...")
		alpsV1Clusts = []
		#cluster and clean vectors
		zthresh = 3.5
		#minDistRadsProtected = 0.2 #if an index exceeds the mean dist zscore threshold, but is under this value, protect it from censoring.
		#this ensures that if vectors are very tight we don't reject otherwise good vecotrs?
		#worst case it rejects a couple though really isn't a big deal. 
		#currently not implemented
		for i in range(4):
			alpsDist = greatCircleDistanceMatrix(np.array(alps_v1s[i]))
			
			maxMeanDist = 100
			min_samples = 5
			max_min_samples = 20
			while (maxMeanDist > 1) and (min_samples <= max_min_samples): #if a vector has a mean distance to other vectors exceeding 1 radian, redo with tighter clustering
				
				alps_v1_clust = DBSCAN(metric='precomputed', min_samples=min_samples).fit(alpsDist)
				alpsv1Dist0s = np.mean(greatCircleDistanceMatrix(np.array(alps_v1s[i])[alps_v1_clust.labels_ == 0,:]), axis=0)
				#print(alpsv1Dist0s)
				maxMeanDist = np.max(alpsv1Dist0s)
				if (maxMeanDist > 1):
					min_samples += 5
					if diagnositcsVecs:
						if (min_samples <= max_min_samples):
							print("-- Vector set",i,"exceeds max mean distance of r = 1, increasing min_samples", min_samples)
						else:
							print("-- Vector set",i,"exceeds max mean distance of r = 1, reached max min_samples, breaking")
			#finally, compute the zscore distances of the remaining vectors, 
			alpsv1Dist0s_m = np.mean(alpsv1Dist0s)
			alpsv1Dist0s_sd = np.std(alpsv1Dist0s)
			z = np.abs((alpsv1Dist0s - alpsv1Dist0s_m) / alpsv1Dist0s_sd)
			#set outliers to -1 label 
			if np.sum(z>zthresh) > 0:
				if diagnositcsVecs:
					print("-- Censoring ", np.sum(z>zthresh), "from vector set", i, "as distance Z exceeds", zthresh)
				idx = 0
				for j in range(len(alps_v1_clust.labels_)):
					if alps_v1_clust.labels_[j] == 0:
						if z[idx] > zthresh:
							alps_v1_clust.labels_[j] = -1
						idx += 1
			del alpsv1Dist0s_m, alpsv1Dist0s_sd, z, maxMeanDist
			#recompute distances
			alpsv1Dist0s = np.mean(greatCircleDistanceMatrix(np.array(alps_v1s[i])[alps_v1_clust.labels_ == 0,:]), axis=0)
			alpsV1Clusts.append(alps_v1_clust)
			temp = np.array(alps_indices[i])
			alpsIndicesClean.append(temp[:, alpsV1Clusts[i].labels_ == 0])
	
			
			alpsv1_median_n0_index = np.where(alpsv1Dist0s == np.min(alpsv1Dist0s))[0][0]
			alpsv1_median = np.array(alps_v1s[i])[alpsV1Clusts[i].labels_ == 0,:][alpsv1_median_n0_index,:]
			alpsMedianV1Clean.append(alpsv1_median)
			if diagnostics:
				plot_quiver4(np.array(alps_v1s[i]).T, 1, "Alps ROI "+str(i+1)+" \n" + \
							 str(np.sum(alps_v1_clust.labels_ == 0)) +" of " +\
							 str(len(alps_v1_clust.labels_)) +" vectors retained\n" + \
							 "Green = retained, Gold = rejected", \
							 alps_v1_clust.labels_)
	else:
		#do not cluster and clean voxels in each ROI,just pass originals
		for i in range(4):
			#for each ROI, copy original indices to clean indices
			alpsIndicesClean.append(np.array(alps_indices[i]))
			#select median vector in the same way, but on all roi voxels
			alpsv1Dist0s = np.mean(greatCircleDistanceMatrix(np.array(alps_v1s[i])), axis=0)
			alpsv1_median_n0_index = np.where(alpsv1Dist0s == np.min(alpsv1Dist0s))[0][0]
			alpsv1_median = np.array(alps_v1s[i])[alpsv1_median_n0_index,:]
			alpsMedianV1Clean.append(alpsv1_median)


	#compute ALPS
	print("- Computing local ALPS metrics")
	newAlpsMetrics = {}
	qc_vox_Xinterp_adc = []
	qc_vox_Iinterp_adc = []
	for i in range(4):
		roi_Xinterp_adcs = []
		roi_Iinterp_adcs = []
		xi, yi, zi = alpsIndicesClean[i]
		for voxIndices in range(len(xi)):
			voxV1vec = V1_data[xi[voxIndices],yi[voxIndices],zi[voxIndices],:]
			voxV1vec = normalize_vector(voxV1vec)
			nearRoiMedianV1 = alpsMedianV1Clean[(1-(i%2))+(i//2)*2]
			nearRoiMedianV1 = normalize_vector(nearRoiMedianV1)
			#print(voxV1vec,nearRoiMedianV1, np.cross(voxV1vec,nearRoiMedianV1))
			#/\ that just goes 1 for 0, 0 for 1, 3 for 2, 2 for 3
			voxNewXvec = np.cross(voxV1vec,nearRoiMedianV1)
			voxNewXvec = normalize_vector(voxNewXvec)
			voxNewIvec = np.cross(voxV1vec,voxNewXvec)
			voxNewIvec = normalize_vector(voxNewIvec)
			#interpolate the ADC from this vector in that voxel
			#vectors defined by the eddy 
			#print(voxNewXvec)
			
			#first computing ADC for newX
			projected_vectors = batch_orthographic_projection(bvecs_no0.T, voxNewXvec)
			interp = CloughTocher2DInterpolator(projected_vectors, adcData[xi[voxIndices],yi[voxIndices],zi[voxIndices],:])
			if diagnostics:
				#I should pass the title in here so that someone else could look at this and have any idea what is going on
				plot_mesh2(projected_vectors,adcData[xi[voxIndices],yi[voxIndices],zi[voxIndices],:] )
			voxNewXadc = interp((0,0))
			
			#now computing ADC for newI
			projected_vectors = batch_orthographic_projection(bvecs_no0.T, voxNewIvec)
			interp = CloughTocher2DInterpolator(projected_vectors, adcData[xi[voxIndices],yi[voxIndices],zi[voxIndices],:])
			if diagnostics:
				#I should pass the title in here so that someone else could look at this and have any idea what is going on
				plot_mesh2(projected_vectors,adcData[xi[voxIndices],yi[voxIndices],zi[voxIndices],:] )
			voxNewIadc = interp((0,0))
			#sanity check
			#do either of the interpolated values exceed the range of input values, or zscore them relative to input set
			roi_Xinterp_adcs.append(voxNewXadc)
			roi_Iinterp_adcs.append(voxNewIadc)
			qc_vox_Xinterp_adc.append(voxNewXadc)
			qc_vox_Iinterp_adc.append(voxNewIadc)
		newAlpsMetrics[roiLabels[i]+"_x"]=np.mean(roi_Xinterp_adcs)
		newAlpsMetrics[roiLabels[i]+"_i"]= np.mean(roi_Iinterp_adcs)


	if diagnosticsHist:
		plt.hist(qc_vox_Xinterp_adc, label="qc_vox_Xinterp_adc")
		plt.hist(qc_vox_Iinterp_adc, label="qc_vox_Iinterp_adc")
		plt.legend()
		plt.title(prefix +sub +"  qc_vox_interp_adc_zscores")
		plt.show()
	print(newAlpsMetrics)

	L_alps_score = np.mean([newAlpsMetrics["L_Association_x"],newAlpsMetrics["L_Projection_x"]]) / np.mean([newAlpsMetrics["L_Association_i"],newAlpsMetrics["L_Projection_i"]])
	R_alps_score = np.mean([newAlpsMetrics["R_Association_x"],newAlpsMetrics["R_Projection_x"]]) / np.mean([newAlpsMetrics["R_Association_i"],newAlpsMetrics["R_Projection_i"]])
	return [np.mean(L_alps_score, R_alps_score), L_alps_score, R_alps_score, newAlpsMetrics["L_Association_x"], newAlpsMetrics["L_Projection_x"], newAlpsMetrics["L_Association_i"], newAlpsMetrics["L_Projection_i"], newAlpsMetrics["R_Association_x"], newAlpsMetrics["R_Projection_x"], newAlpsMetrics["R_Association_i"], newAlpsMetrics["R_Projection_i"]]

#%%



subs = []
allItemsInBaseDir = os.listdir(baseDir)
for item in allItemsInBaseDir:
	#identify alps directories
	if os.path.isdir(os.path.join(baseDir, item)):
		if item.startswith("alps_"):
			subs.append(item)


data = {}
for sub in subs:
	try:
		data[sub] = alpsCalc('', sub)
	except:
		pass

#%%
print("File, alps, L_alps, R_alps, L_Association_x, L_Projection_x, L_Association_i, L_Projection_i, R_Association_x, R_Projection_x, R_Association_i, R_Projection_i")
for datum in data:
	print(datum, data[datum])

