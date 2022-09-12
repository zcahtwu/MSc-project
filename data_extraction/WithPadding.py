################### UNEQUAL, MIX , 32 ###########

# Save images for both acceptale classes and unaccetable classes by following protocols as follows:

# 0. Shuffle data/volumes. Divide data in three parts - train , test valid. Division is for samples of both classes  
#    such that imbalance ratio issame for mentioned three divisions. 
# 1. Scale each 3D Volume
# 2. Generate the csv files provifing names of saved 2D slices and the labels for each 2D slices. 


# 3. Extract central 32 slices.

import os
import numpy as np
import nibabel as nib
import csv
import pandas as pd
# import imageio
import h5py
path='/media/prabhubuntu/20C6752CC67502F8/data/4_2_ReplicateSujitResized'
import matplotlib.pyplot as plt
import math

csvre=pd.read_csv('abide.csv',sep=',',index_col=False)
labelsCSV = pd.DataFrame(csvre.values)
labelsCSV=labelsCSV.sample(n=len(csvre),random_state=5)  # Mix yeah !
lists=list(labelsCSV[2])
labelsCSV.to_csv('randomized.csv')

cnt=0
labels1=[]   	
labels2=[]
finallabel=[]
'''
for foldername in os.listdir(path):
	foldername.replace(path,'')
	print(foldername)
	row=csvre.loc[csvre["filenames"]==foldername]
	if not(row.empty):
		#print(row)
		labels1.append((row.iloc[0]["qc_anat_rater_2"]))
		labels2.append(row.iloc[0]["qc_anat_rater_3"])
		#print(labels1)
		#print(labels2)
		if labels1[cnt]=="fail" or labels2[cnt]=="fail":
			finallabel.append(1)
		else:
			finallabel.append(0)
		cnt+=1

print(finallabel)
'''

sagFilenames=[]
corFilenames=[]
Filenames=[]
cor_timages=[]
ax_timages=[]
timages=[]
tlabels=[]
failed=[]
cntOne=0
cntZero=0

total=len(lists)
total1=133# Choose 133 when both raters ratings are considered, 98 is failed by only rater 3
total0=total-total1
print('Total volumes, good and bad volumes:',total,total0,total1)
name='New_Axial'


#-----------------------------------------------------------------------------------------------------

cnnt=0
maxx1=np.round(total1*0.6)-1# 133*0.6=79.8=80
maxx0=np.round(total0*0.6)-1# 966*0.6=580, ~1:7 ratio
cnt0=0
cnt1=0


print('Max ones and zeros samples are:', maxx1,maxx0)


hf = h5py.File('AbideTrain'+name+'.h5', 'w')

img=hf.create_dataset('image', (32,256,256),chunks=True,maxshape=(None,256,256))
lab=hf.create_dataset('label', (32,), chunks=True,maxshape=(None,))

for i in range(4,len(labelsCSV)):
	try:
		names=labelsCSV.iloc[i][1]#+".nii"
		print(names)
		filename=labelsCSV.iloc[i][3]#os.sep.join([path,names])
		imgdata=nib.load(filename)
		imgs=imgdata.get_fdata()
		#print(imgs.any())
	except OSError as error:
		print('Error: failedtoRead')
		print(filename)
	else:
		label=int(labelsCSV.iloc[i][2])
		
		if (label==1 and cnt1<maxx1) or (label==0 and cnt0<maxx0):
			if label==1:
				cnt1+=1
			if int(labelsCSV.iloc[i][2])==0:
				cnt0+=1
			#print('train',cnt1,cnt0)
			imgs=(imgs-np.amin(imgs))/(np.amax(imgs)-np.amin(imgs))#SCALE IN [0 1]
			ids=np.arange(48, 208, 5).tolist() #SELECT CENTRAL 32 SLICES
			#print(filename,labelsCSV.iloc[i][2])
			# Compute labels
			with open('csvs/AbideTrain'+name+'.csv', 'a') as f:
				csvwriter = csv.writer(f) 
				csvwriter.writerow([labelsCSV.iloc[i][1],labelsCSV.iloc[i][2]])

			## PADDING PORTION
			volshape=np.shape(imgs)
			vol = np.zeros([256,256,256])
			print(volshape)
			if volshape[0]<=256:
				# compute center offset
				z_center = math.floor((256-volshape[0]) / 2)
				
				# copy img image into center of result image
				#vol[z_center:z_center+volshape[0],:,:] = imgs
				flag0=0
			else:
				
				# compute center offset
				z_center = math.floor((volshape[0]-256) / 2)
				# copy img image into center of result image
				#vol=imgs[z_center:z_center+volshape[0],:,:]
				flag0=1
			
			
			
			if volshape[1]<=256:

				
				# compute center offset
				y_center = math.floor((256-volshape[1]) / 2)

				flag1=0
				# copy img image into center of result image

				'''
				if flag0==0:
					vol[z_center:z_center+volshape[0],y_center:y_center+volshape[1],:] = imgs
				else:
					vol[:,y_center:y_center+volshape[1],:] = imgs[z_center:z_center+volshape[0],:,:]
				'''

			else:
				
				# compute center offset
				y_center = math.floor((volshape[1]-256) / 2)
				flag1=1
				# copy img image into center of result image
				
			
			
			if volshape[2]<=256:
				
				# compute center offset
				x_center = math.floor((256-volshape[2]) / 2)
				# copy img image into center of result image
				
				if flag0==0 and flag1==0:
					vol[z_center:z_center+volshape[0], y_center:y_center+volshape[1],x_center:x_center+volshape[2]] = imgs
				else:
					if flag0==0 and flag1==1:
						vol[z_center:z_center+volshape[0], :,x_center:x_center+volshape[2]] = imgs[:,y_center:-y_center+volshape[1],:]
					else:
						if flag0==1 and flag1==0:
							vol[:, y_center:y_center+volshape[1],x_center:x_center+volshape[2]] = imgs[z_center:-z_center+volshape[0],:,:]
						else:
							if flag0==1 and flag1==1:
								vol[:, :,x_center:x_center+volshape[2]] = imgs[z_center:-z_center+volshape[0],y_center:-y_center+volshape[1],:]
					
			else:
				
				# compute center offset
				x_center = (volshape[2]-256) // 2
				# copy img image into center of result image
				if flag0==0 and flag1==0:
					vol[z_center:z_center+volshape[0], y_center:y_center+volshape[1],:] = imgs[:,:,x_center:x_center+256]
				else:
					if flag0==0 and flag1==1:
						vol[z_center:z_center+volshape[0], :,:] = imgs[:,y_center:y_center+256,x_center:x_center+256]
					else:
						if flag0==1 and flag1==0:
							vol[:, y_center:y_center+volshape[1],:] = imgs[z_center:z_center+256,:,x_center:x_center+256]
						else:
							if flag0==1 and flag1==1:
								vol[:, :,:] = imgs[z_center:z_center+256,y_center:y_center+256,x_center:x_center+256]
			
			
			print(np.shape(vol))
			imgs=vol
			print(np.shape(imgs))
			
			# plt.figure(figsize=(30,30))


			for j in range(len(ids)):
				temp=imgs[ids[j]] # temp is image in sag plane
				ax_temp=imgs[:,:,ids[j]] # ax_temp is image in axial plane
				ax_temp=np.rot90(ax_temp)
				
				
				#matplotlib.image.imsave('name.png', ax_temp)
				# plt.subplot(6,6,j+1)
				# plt.xticks([])
				# plt.yticks([])
				# plt.grid(False)
				# im=255*ax_temp
				# plt.imshow(im,cmap='gray')
				# plt.xlabel(str(label))
				
				try:
					if j==0:
						images2=np.expand_dims(ax_temp,axis=0)
					else:
						images2=np.concatenate((images2,np.expand_dims(ax_temp,axis=0)), axis=0)
					tlabels.append(int(labelsCSV.iloc[i][2])) # int because labelscsv has labels as strings ' 1' or ' 0'
						#print(images2.any())
				except OSError as error: 
					failed.append(labelsCSV[0][i])


			# plt.savefig('Axial/Train/'+names+'_label'+str(label)+'.png')
			# plt.close()

			if not(images2.any()):
				print('zeros here')
				print(cnnt)
				print(i)
			if cnnt!=0:
				img.resize((cnnt+1)*32,axis=0)
				lab.resize((cnnt+1)*32,axis=0)

			img[cnnt*32:(cnnt*32)+32]=images2
			lab[cnnt*32:(cnnt*32)+32]=np.asarray(tlabels)
			tlabels=[]
			cnnt+=1
print(failed)
hf.close()
print('samples scanned for ones and zeros are:',cnt1,cnt0)



#-----------------------------------------------------------------------------------------------------
hf = h5py.File('AbideValid'+name+'.h5', 'w')
img=hf.create_dataset('image', (32,256,256),chunks=True,maxshape=(None,256,256))
lab=hf.create_dataset('label', (32,), chunks=True,maxshape=(None,))


cnnt=0
maxx1=np.round(total1*0.6) 
maxx0=np.round(total0*0.6)
cnt0=0
cnt1=0
maxxv1=maxx1+(np.round(total1*0.2))
maxxv0=maxx0+(np.round(total0*0.2))


print('Valid: Max ones samples are:', maxx1, ' to ', maxxv1)
print('Valid: Max zeros samples are:', maxx0, ' to ', maxxv0)
for i in range(len(labelsCSV)):
	try:
		names=labelsCSV.iloc[i][1]+".nii"
		filename=os.sep.join([path,names])
		imgdata=nib.load(filename)
		imgs=imgdata.get_fdata()
	except OSError as error:
		print('Error: failedtoRead')
		print(filename)
	else:
		label=int(labelsCSV.iloc[i][2])
		if label==1 and cnt1<maxx1:
			cnt1+=1
		if label==0 and cnt0<maxx0:
			cnt0+=1
		if (label==1 and cnt1>=maxx1 and cnt1<maxxv1) or (label==0 and cnt0>=maxx0 and cnt0<maxxv0):
			if label==1:
				cnt1+=1
			if int(labelsCSV.iloc[i][2])==0:
				cnt0+=1
			#print('valid',cnt1,cnt0)
			imgs=(imgs-np.amin(imgs))/(np.amax(imgs)-np.amin(imgs))#SCALE IN [0 1]
			ids=np.arange(48, 208, 5).tolist() #SELECT CENTRAL 32 SLICES
			#print(filename,labelsCSV.iloc[i][2])
			# Compute labels
			with open('csvs/AbideValid'+name+'.csv', 'a') as f:
				csvwriter = csv.writer(f) 
				csvwriter.writerow([labelsCSV.iloc[i][1],labelsCSV.iloc[i][2]])
			
			
			## PADDING PORTION
			volshape=np.shape(imgs)
			vol = np.zeros([256,256,256])
			print(volshape)
			
			if volshape[0]<=256:
				# compute center offset
				z_center = math.floor((256-volshape[0]) / 2)
				
				# copy img image into center of result image
				#vol[z_center:z_center+volshape[0],:,:] = imgs
				flag0=0
			else:
				
				# compute center offset
				z_center = math.floor((volshape[0]-256) / 2)
				# copy img image into center of result image
				#vol=imgs[z_center:z_center+volshape[0],:,:]
				flag0=1
			
			
			
			if volshape[1]<=256:

				
				# compute center offset
				y_center = math.floor((256-volshape[1]) / 2)

				flag1=0
				# copy img image into center of result image
				'''if flag0==0:
					vol[z_center:z_center+volshape[0],y_center:y_center+volshape[1],:] = imgs
				else:
					vol[:,y_center:y_center+volshape[1],:] = imgs[z_center:z_center+volshape[0],:,:]'''
			else:
				
				# compute center offset
				y_center = math.floor((volshape[1]-256) / 2)
				flag1=1
				# copy img image into center of result image
				
			
			
			if volshape[2]<=256:
				
				# compute center offset
				x_center = math.floor((256-volshape[2]) / 2)
				# copy img image into center of result image
				
				if flag0==0 and flag1==0:
					vol[z_center:z_center+volshape[0], y_center:y_center+volshape[1],x_center:x_center+volshape[2]] = imgs
				else:
					if flag0==0 and flag1==1:
						vol[z_center:z_center+volshape[0], :,x_center:x_center+volshape[2]] = imgs[:,y_center:-y_center+volshape[1],:]
					else:
						if flag0==1 and flag1==0:
							vol[:, y_center:y_center+volshape[1],x_center:x_center+volshape[2]] = imgs[z_center:-z_center+volshape[0],:,:]
						else:
							if flag0==1 and flag1==1:
								vol[:, :,x_center:x_center+volshape[2]] = imgs[z_center:-z_center+volshape[0],y_center:-y_center+volshape[1],:]
					
			else:
				
				# compute center offset
				x_center = (volshape[2]-256) // 2
				# copy img image into center of result image
				if flag0==0 and flag1==0:
					vol[z_center:z_center+volshape[0], y_center:y_center+volshape[1],:] = imgs[:,:,x_center:x_center+256]
				else:
					if flag0==0 and flag1==1:
						vol[z_center:z_center+volshape[0], :,:] = imgs[:,y_center:y_center+256,x_center:x_center+256]
					else:
						if flag0==1 and flag1==0:
							vol[:, y_center:y_center+volshape[1],:] = imgs[z_center:z_center+256,:,x_center:x_center+256]
						else:
							if flag0==1 and flag1==1:
								vol[:, :,:] = imgs[z_center:z_center+256,y_center:y_center+256,x_center:x_center+256]
			
			
			plt.figure(figsize=(30,30))
			
			for j in range(len(ids)):
				temp=imgs[ids[j]] # temp is image in sag plane
				ax_temp=imgs[:,:,ids[j]] # ax_temp is image in axial plane
				ax_temp=np.rot90(ax_temp)
				shapes=np.shape(ax_temp)
				
				if shapes[0]<=256 and shapes[1]<=256:
					result = np.zeros([256,256])

					# compute center offset
					x_center = (256 - shapes[0]) // 2
					y_center = (256 - shapes[1]) // 2

					# copy img image into center of result image
					result[x_center:x_center+shapes[0],y_center:y_center+shapes[1]] = ax_temp
				else:
					result = np.zeros([256,256])

					# compute center offset
					if shapes[0]>256:
						x_center = (shapes[0]-256) // 2
					else:
						x_center = (256 - shapes[0]) // 2
					if  shapes[1]>256:
						y_center = (shapes[1]-256) // 2
					else:
						y_center = (256 - shapes[1]) // 2
					# copy img image into center of result image
					result=ax_temp[x_center:x_center+shapes[0],y_center:y_center+shapes[1]]
				
				ax_temp=result
				print(np.shape(ax_temp))
							
				plt.subplot(6,6,j+1)
				plt.xticks([])
				plt.yticks([])
				plt.grid(False)
				im=255*ax_temp
				plt.imshow(im,cmap='gray')
				plt.xlabel(str(label))
				
				
				try:
					if j==0:
						images2=np.expand_dims(ax_temp,axis=0)
					else:
						images2=np.concatenate((images2,np.expand_dims(ax_temp,axis=0)),axis=0)

					tlabels.append(int(labelsCSV.iloc[i][2])) # int because labelscsv has labels as strings ' 1' or ' 0'
					
				except OSError as error: 
					failed.append(labelsCSV[0][i])

			plt.savefig('Axial/Valid/'+names[:-4]+'_label'+str(label)+'.png')
			plt.close()
			if not(images2.any()):
				print('zeros here')
				print(cnnt)
				print(i)
			if cnnt!=0:
				img.resize((cnnt+1)*32,axis=0)
				lab.resize((cnnt+1)*32,axis=0)

			img[cnnt*32:(cnnt*32)+32]=images2
			lab[cnnt*32:(cnnt*32)+32]=np.asarray(tlabels)
			tlabels=[]
			cnnt+=1
print(failed)



#hf.create_dataset('subjectID', data=np.asarray(Filenames))
hf.close()

print('samples scanned for ones and zeros are:',cnt1,cnt0)

#----------------------------------------------------------------------------------------------------------
#hf = h5py.File('/scratch0/prabkaur/data/train_accelerated_MixGo23OkFail1by7Final.h5', 'w') # maxx=1045
hf = h5py.File('AbideTest'+name+'.h5', 'w')

img=hf.create_dataset('image', (32,256,256),chunks=True,maxshape=(None,256,256))
lab=hf.create_dataset('label', (32,), chunks=True,maxshape=(None,))


cnnt=0
maxx1=np.round(total1*0.6) # 0.6*115
maxx0=np.round(total0*0.6)
cnt0=0
cnt1=0
maxxv1=maxx1+(np.round(total1*0.2)) # 0.6*115
maxxv0=maxx0+(np.round(total0*0.2)) # 0.6*115

maxxts1=total1#maxxv1+(np.round(total1*0.2))
maxxts0=total0#(np.round(total0*0.2))+maxxv0

print('Valid: Max ones samples are:', maxxv1, ' to ', maxxts1)
print('Valid: Max zeros samples are:', maxxv0, ' to ', maxxts0)

for i in range(len(labelsCSV)):
	try:
		names=labelsCSV.iloc[i][1]+".nii"
		filename=os.sep.join([path,names])
		imgdata=nib.load(filename)
		imgs=imgdata.get_fdata()
	except OSError as error:
		print('Error: failedtoRead')
		print(filename)
	else:
		label=int(labelsCSV.iloc[i][2])
		if label==1 and cnt1<maxxv1:
			cnt1+=1
		if label==0 and cnt0<maxxv0:
			cnt0+=1
		if (label==1 and cnt1>=maxxv1 and cnt1<=maxxts1) or (label==0 and cnt0>=maxxv0 and cnt0<=maxxts0):
			if label==1:
				cnt1+=1
			if int(labelsCSV.iloc[i][2])==0:
				cnt0+=1
			print('test',cnt1,cnt0)
			imgs=(imgs-np.amin(imgs))/(np.amax(imgs)-np.amin(imgs))#SCALE IN [0 1]
			ids=np.arange(48, 208, 5).tolist() #SELECT CENTRAL 32 SLICES
			#print(filename,labelsCSV.iloc[i][2])
			# Compute labels
			with open('csvs/AbideTest'+name+'.csv', 'a') as f:
				csvwriter = csv.writer(f) 
				csvwriter.writerow([labelsCSV.iloc[i][1],labelsCSV.iloc[i][2]])
			
			
			## PADDING PORTION
			volshape=np.shape(imgs)
			vol = np.zeros([256,256,256])
			print(volshape)
			if volshape[0]<=256:
				# compute center offset
				z_center = math.floor((256-volshape[0]) / 2)
				
				# copy img image into center of result image
				#vol[z_center:z_center+volshape[0],:,:] = imgs
				flag0=0
			else:
				
				# compute center offset
				z_center = math.floor((volshape[0]-256) / 2)
				# copy img image into center of result image
				#vol=imgs[z_center:z_center+volshape[0],:,:]
				flag0=1
			
			
			
			if volshape[1]<=256:

				
				# compute center offset
				y_center = math.floor((256-volshape[1]) / 2)

				flag1=0
				# copy img image into center of result image
				'''if flag0==0:
					vol[z_center:z_center+volshape[0],y_center:y_center+volshape[1],:] = imgs
				else:
					vol[:,y_center:y_center+volshape[1],:] = imgs[z_center:z_center+volshape[0],:,:]'''
			else:
				
				# compute center offset
				y_center = math.floor((volshape[1]-256) / 2)
				flag1=1
				# copy img image into center of result image
				
			
			
			if volshape[2]<=256:
				
				# compute center offset
				x_center = math.floor((256-volshape[2]) / 2)
				# copy img image into center of result image
				
				if flag0==0 and flag1==0:
					vol[z_center:z_center+volshape[0], y_center:y_center+volshape[1],x_center:x_center+volshape[2]] = imgs
				else:
					if flag0==0 and flag1==1:
						vol[z_center:z_center+volshape[0], :,x_center:x_center+volshape[2]] = imgs[:,y_center:-y_center+volshape[1],:]
					else:
						if flag0==1 and flag1==0:
							vol[:, y_center:y_center+volshape[1],x_center:x_center+volshape[2]] = imgs[z_center:-z_center+volshape[0],:,:]
						else:
							if flag0==1 and flag1==1:
								vol[:, :,x_center:x_center+volshape[2]] = imgs[z_center:-z_center+volshape[0],y_center:-y_center+volshape[1],:]
					
			else:
				
				# compute center offset
				x_center = (volshape[2]-256) // 2
				# copy img image into center of result image
				if flag0==0 and flag1==0:
					vol[z_center:z_center+volshape[0], y_center:y_center+volshape[1],:] = imgs[:,:,x_center:x_center+256]
				else:
					if flag0==0 and flag1==1:
						vol[z_center:z_center+volshape[0], :,:] = imgs[:,y_center:y_center+256,x_center:x_center+256]
					else:
						if flag0==1 and flag1==0:
							vol[:, y_center:y_center+volshape[1],:] = imgs[z_center:z_center+256,:,x_center:x_center+256]
						else:
							if flag0==1 and flag1==1:
								vol[:, :,:] = imgs[z_center:z_center+256,y_center:y_center+256,x_center:x_center+256]
			
			
			
			plt.figure(figsize=(30,30))
						
			for j in range(len(ids)):
				temp=imgs[ids[j]] # temp is image in sag plane
				ax_temp=imgs[:,:,ids[j]] # ax_temp is image in axial plane
				ax_temp=np.rot90(ax_temp)
				shapes=np.shape(ax_temp)
				
				if shapes[0]<=256 and shapes[1]<=256:
					result = np.zeros([256,256])

					# compute center offset
					x_center = (256 - shapes[0]) // 2
					y_center = (256 - shapes[1]) // 2

					# copy img image into center of result image
					result[x_center:x_center+shapes[0],y_center:y_center+shapes[1]] = ax_temp
				else:
					result = np.zeros([256,256])

					# compute center offset
					if shapes[0]>256:
						x_center = (shapes[0]-256) // 2
					else:
						x_center = (256 - shapes[0]) // 2
					if  shapes[1]>256:
						y_center = (shapes[1]-256) // 2
					else:
						y_center = (256 - shapes[1]) // 2
					# copy img image into center of result image
					result=ax_temp[x_center:x_center+shapes[0],y_center:y_center+shapes[1]]
				
				ax_temp=result
				print(np.shape(ax_temp))
				plt.subplot(6,6,j+1)
				plt.xticks([])
				plt.yticks([])
				plt.grid(False)
				im=255*ax_temp
				plt.imshow(im,cmap='gray')
				plt.xlabel(str(label))
				
				try:
					if j==0:
						images2=np.expand_dims(ax_temp,axis=0)
					else:
						images2=np.concatenate((images2,np.expand_dims(ax_temp,axis=0)),axis=0)

					tlabels.append(int(labelsCSV.iloc[i][2])) # int because labelscsv has labels as strings ' 1' or ' 0'
					
				except OSError as error: 
					failed.append(labelsCSV[0][i])
					
			plt.savefig('Axial/Test/'+names[:-4]+'_label'+str(label)+'.png')
			plt.close()
			if not(images2.any()):
				print('zeros here')
				print(cnnt)
				print(i)
			if cnnt!=0:
				img.resize((cnnt+1)*32,axis=0)
				lab.resize((cnnt+1)*32,axis=0)

			img[cnnt*32:(cnnt*32)+32]=images2
			lab[cnnt*32:(cnnt*32)+32]=np.asarray(tlabels)
			tlabels=[]
			cnnt+=1
print(failed)
print('scans used for ones and zeros are:',cnt1,cnt0)


#hf.create_dataset('subjectID', data=np.asarray(Filenames))
hf.close()
#np.savez(os.sep.join([targetpath,'train.npy']),image=images2,labels=np.asarray(tlabels),SubjectID=np.asarray(Filenames))
#np.save(os.sep.join([targetpath,'ScanIds.npy']),np.asarray(Filenames))
#np.save(os.sep.join([targetpath,'labels.npy']),np.asarray(tlabels))


#----------------------------------------------------------------------------------------------------------
