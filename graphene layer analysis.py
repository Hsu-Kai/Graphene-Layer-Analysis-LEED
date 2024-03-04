import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy.ndimage.measurements import center_of_mass
import scipy.optimize as opt

image = skimage.io.imread(fname="Beam Energy 100.0.jpg")
skimage.io.imsave(fname="Beam Energy 100.0.png", arr=image)
image = skimage.io.imread(fname="Beam Energy 100.0.png")
imager = skimage.io.imread(fname="Beam Energy 100.0.png")

# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.xlabel("Color value")
plt.ylabel("Pixels")
plt.show()


imagec = skimage.io.imread(fname="Beam Energy 100.0.png")
imagec[imagec < 30] = 0
skimage.io.imsave(fname="contrast_adjusted Beam Energy 100.0.png", arr=imagec)


green=image[:, :, 1]
greenc=imagec[:, :, 1]

sig = 5
greenf = gaussian_filter(green, sigma=sig, mode='mirror', order=0) # filter/smooth
greencf = gaussian_filter(greenc, sigma=sig, mode='mirror', order=0) # contrast/filter/smooth


skimage.io.imsave(fname="Green_Beam Energy 100.0.png", arr=green)
skimage.io.imsave(fname="Green_filtered Beam Energy 100.0.png", arr=greenf)
skimage.io.imsave(fname="Green_filtered_contrast_adjusted Beam Energy 100.0.png", arr=greencf)

image = skimage.io.imread(fname="Green_filtered Beam Energy 100.0.png", as_gray=True)
imagec = skimage.io.imread(fname="contrast_adjusted Beam Energy 100.0.png", as_gray=True)



blob_log = blob_log(greenc, min_sigma=3.0, max_sigma=11.5, 
num_sigma=10, threshold=.025, overlap=0.0001)

yb, xb, sizes = blob_log.T.copy()

fig, ax = plt.subplots(1, 1)

ax.imshow(greencf, cmap='gray')
ax.plot(xb, yb, 'o', color='none', markeredgecolor='red', 
	markersize=14)
annoff=40
fs=14
rot=0
for i, (x, y, sizes) in enumerate(zip(xb, yb, sizes)):
    ax.annotate(str(i), [x+annoff, y], color='r'
		,fontsize=fs, rotation=rot)
    print(i,(y, x, sizes))
plt.show()

SiClocations = blob_log[[2,7,8,9,11,19],:2] ###check these      #10,11,12,13,15,61 imagec[imagec < 10] = 0
                                                                       #blob_log = blob_log(greenc, min_sigma=3.0, max_sigma=11.5,num_sigma=10, threshold=.020, overlap=0.0001)
BLGlocations = blob_log[[0,1,3,4,5,6],:2] ###check these      #0,1,2,3,5,6 imagec[imagec < 10] = 0
                                                                       #blob_log = blob_log(greenc, min_sigma=3.0, max_sigma=11.5,num_sigma=10, threshold=.020, overlap=0.0001)



### Remove the spots affected by the LEED electron gun and close to the lowest boundary
SiClocations = blob_log[[2,8,9,11,19],:2] ###check these      #10,11,12,13,15,61 imagec[imagec < 10] = 0
                                                                       #blob_log = blob_log(greenc, min_sigma=3.0, max_sigma=11.5,num_sigma=10, threshold=.020, overlap=0.0001)
BLGlocations = blob_log[[0,1,3,5,6],:2] ###check these      #0,1,2,3,5,6 imagec[imagec < 10] = 0
                                                                       #blob_log = blob_log(greenc, min_sigma=3.0, max_sigma=11.5,num_sigma=10, threshold=.020, overlap=0.0001)



SiClocations = SiClocations.round().astype('int32')
BLGlocations = BLGlocations.round().astype('int32')                                                         

hw = 8
BLGspots = []
for BLGy, BLGx in BLGlocations:
    mask1 = np.ones_like(greencf, dtype=bool)
    mask1[BLGy-hw:BLGy+hw+1, BLGx-hw:BLGx+hw+1] = False
    ma1 = np.ma.array(greenf, mask=mask1)
    BLGspots.append(center_of_mass(ma1))

BLGspots = np.array(BLGspots).round().astype('int32')


hw = 8
SiCspots = []
for SiCy, SiCx in SiClocations:
    mask1 = np.ones_like(greencf, dtype=bool)
    mask1[SiCy-hw:SiCy+hw+1, SiCx-hw:SiCx+hw+1] = False
    ma1 = np.ma.array(greenf, mask=mask1)
    SiCspots.append(center_of_mass(ma1))

SiCspots = np.array(SiCspots).round().astype('int32')





fig = plt.figure(figsize=[10.5, 15])
    
plt.subplot(1, 1, 1)
plt.imshow(greenf, cmap='gray')
y1, x1 = BLGspots.T
plt.plot(x1, y1, 'o', color='none', markeredgecolor='red', markersize=10)

y2, x2 = SiCspots.T
plt.plot(x2, y2, 'o', color='none', markeredgecolor='red', markersize=10)

plt.show()   




#Checking position-dependent background at BLG spots & SiC spots
hw2 = 50
fig = plt.figure(figsize=[10.5, 15])
plt.subplot(2, 1, 1)
plt.imshow(greenf, cmap='gray')
y1, x1 = BLGspots.T
plt.plot(x1, y1, 'o', color='none', markeredgecolor='red', markersize=15)
y2, x2 = SiCspots.T
plt.plot(x2, y2, 'o', color='none', markeredgecolor='red', markersize=15)


for i, (BLGy, BLGx) in enumerate(BLGspots):
    plt.subplot(4, 6, i+13)
    cropi = image[BLGy-hw2:BLGy+hw2, BLGx-hw2:BLGx+hw2]
    plt.imshow(cropi, cmap='gray')
    h_mid = cropi[:, 40:61].sum(axis=1)
    v_mid = cropi[40:61,:].sum(axis=0)
    plt.plot(range(100), 100-0.03*h_mid.astype(np.float64), '-r')
    plt.plot(0.03*v_mid.astype(np.float64), range(100), '-g')

for j, (SiCy, SiCx) in enumerate(SiCspots):
    plt.subplot(4, 6, j+19)
    cropi = image[SiCy-hw2:SiCy+hw2, SiCx-hw2:SiCx+hw2]
    plt.imshow(cropi, cmap='gray')
    h_mid = cropi[:, 40:61].sum(axis=1)
    v_mid = cropi[40:61,:].sum(axis=0)
    plt.plot(range(100), 100-0.03*h_mid.astype(np.float64), '-r')
    plt.plot(0.03*v_mid.astype(np.float64), range(100), '-g')
    
plt.show()



# Define Gaussian Function for Fitting 
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x,y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


#Generate Data For Fitting BLG & SiC spots
x = np.linspace(0, 99, 100)                                # Create x and y indices
y = np.linspace(0, 99, 100)
x, y = np.meshgrid(x, y)
xdata_tuple = (x,y)
data = twoD_Gaussian(xdata_tuple, 3, 50, 50, 20, 20, 0, 5) #create data & trial parameters:(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)

plt.figure()                                               # plot twoD_Gaussian data generated above
plt.imshow(data.reshape(100, 100))
plt.colorbar()
plt.show()


a_BLG_opt = []
intensity_BLG = []
#Gaussian Fitting for BLG spots
for i, (BLGy, BLGx) in enumerate(BLGspots):
    plt.subplot(3, 6, i+1)
    cropi = image[BLGy-hw2:BLGy+hw2, BLGx-hw2:BLGx+hw2]
    plt.imshow(cropi, cmap='gray', origin='lower')
    
    plt.subplot(3, 6, i+7)
    initial_guess = (3, 50, 50, 20, 20, 0, 5)
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata_tuple, cropi.reshape(10000), p0=initial_guess)
    data_fitted = twoD_Gaussian(xdata_tuple, *popt)
    print(popt)
    plt.imshow(cropi.reshape(100, 100), cmap=plt.cm.jet, origin='lower')
    plt.contour(x, y, data_fitted.reshape(100, 100), 8, colors='w', origin='lower')

    plt.subplot(3, 6, i+13)

    
    Y_sub = data_fitted.reshape(100, 100) - np.ones_like(data_fitted.reshape(100, 100), dtype=np.float64)*popt[6]
    a_BLG_opt.append(Y_sub)
    plt.imshow(Y_sub, cmap=plt.cm.jet, origin='lower')
    intensity = (image[BLGy, BLGx].astype(np.float64)+image[BLGy-1, BLGx].astype(np.float64)+image[BLGy+1, BLGx].astype(np.float64)
                 +image[BLGy, BLGx-1].astype(np.float64)+image[BLGy, BLGx+1].astype(np.float64))-5*popt[6]
    print('i, BLGy, BLGx, intensity:', i, BLGy, BLGx, intensity)
    intensity_BLG.append(intensity)
plt.show()


a_SiC_opt = []
intensity_SiC = []
#Gaussian Fitting for SiC spots
for i, (SiCy, SiCx) in enumerate(SiCspots):
    plt.subplot(3, 6, i+1)
    cropi = image[SiCy-hw2:SiCy+hw2, SiCx-hw2:SiCx+hw2]
    plt.imshow(cropi, cmap='gray', origin='lower')
    
    plt.subplot(3, 6, i+7)
    initial_guess = (3, 50, 50, 20, 20, 0, 5)
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata_tuple, cropi.reshape(10000), p0=initial_guess)
    data_fitted = twoD_Gaussian(xdata_tuple, *popt)
    print(popt)
    plt.imshow(cropi.reshape(100, 100), cmap=plt.cm.jet, origin='lower')
    plt.contour(x, y, data_fitted.reshape(100, 100), 8, colors='w', origin='lower')

    plt.subplot(3, 6, i+13)
    Y_sub = data_fitted.reshape(100, 100) - np.ones_like(data_fitted.reshape(100, 100), dtype=np.float64)*popt[6]
    a_SiC_opt.append(Y_sub)
    plt.imshow(Y_sub, cmap=plt.cm.jet, origin='lower')
    intensity = (image[SiCy, SiCx].astype(np.float64)+image[SiCy-1, SiCx].astype(np.float64)+image[SiCy+1, SiCx].astype(np.float64)
                 +image[SiCy, SiCx-1].astype(np.float64)+image[SiCy, SiCx+1].astype(np.float64))-5*popt[6]
    print('i, SiCy, SiCx, intensity: ', i, SiCy, SiCx, intensity)
    intensity_SiC.append(intensity)    
plt.show()


a_BLG_opt = np.array(a_BLG_opt)
a_SiC_opt = np.array(a_SiC_opt)

intensity_BLG = np.array(intensity_BLG)
intensity_SiC = np.array(intensity_SiC)

intensity_sum_BLG = intensity_BLG.sum()
intensity_sum_SiC = intensity_SiC.sum()
ratio=(intensity_sum_BLG)/(intensity_sum_SiC)
print('ratio: ', ratio)   #ratio: 1.9021681637096586


f=np.linspace(0,1,100)
I=0.24                  #I=I1/I0
beta=0.29
y1=f*0.24/((1-f)+0.29*f)
y2=0.24*(1+0.29*f)/((1-f)*0.29+(0.29**2)*f)
y3=0.24*(1+0.29+(0.29**2)*f)/((1-f)*(0.29**2)+(0.29**3)*f)
y4=0.24*(1+0.29+(0.29*2)+(0.29**3)*f)/((1-f)*(0.29**3)+(0.29**4)*f)

if 0 < ratio <= 0.8275862068965517:
    print('graphene thickness: ', np.interp(ratio, y1, f), 'layer(s)')
elif 0.8275862068965517 < ratio <= 3.681331747919144:
    print('graphene thickness: ', 1+np.interp(ratio, y2, f), 'layer(s)')
elif 3.681331747919144 < ratio <= 13.521833613514291:
    print('graphene thickness: ', 2+np.interp(ratio, y3, f), 'layer(s)')
else:
    print('graphene thickness: ', 3+np.interp(ratio, y4, f), 'layer(s)')
    
    #graphene thickness:   1.6755943696383713 layer(s)
