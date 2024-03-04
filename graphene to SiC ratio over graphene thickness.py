import numpy as np 
import matplotlib.pyplot as plt


if True:
    f=np.linspace(0,1,100)
    I=0.24                  #I=I1/I0
    beta=0.29
    y1=f*0.24/((1-f)+0.29*f)
    y2=0.24*(1+0.29*f)/((1-f)*0.29+0.29*0.29*f)
    y3=0.24*(1+0.29+0.29*0.29*f)/((1-f)*0.29*0.29+0.29*0.29*0.29*f)
    y4=0.24*(1+0.29+0.29*0.29+0.29*0.29*0.29*f)/((1-f)*0.29*0.29*0.29+0.29*0.29*0.29*0.29*f)
    
    plt.plot(f,y1,1+f,y2,2+f,y3)
    plt.xticks(fontsize= 16)
    plt.yticks(fontsize= 16)
    plt.xlabel('Graphene Thickness(ML)', loc ="center", fontsize=18)
    plt.ylabel('Graphene:SiC LEED Intensity Ratio', loc ="center", fontsize=18)
    plt.xlim([0.1, 2.5])
    plt.ylim([0.025, 6])
    plt.xticks([0.1,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5],['0.1','0.5','0.75','1.0','1.25','1.5','1.75','2.0','2.25','2.5'])
    plt.yticks([0.025,0.5,1,2,3,5],['0.025','0.5','1','2','3','5']) 
    plt.show()



if True:
    f=np.linspace(0,1,100)
    I=0.24                  #I=I1/I0
    beta=0.29
    y1=f*0.24/((1-f)+0.29*f)
    y2=0.24*(1+0.29*f)/((1-f)*0.29+0.29*0.29*f)
    y3=0.24*(1+0.29+0.29*0.29*f)/((1-f)*0.29*0.29+0.29*0.29*0.29*f)
    y4=0.24*(1+0.29+0.29*0.29+0.29*0.29*0.29*f)/((1-f)*0.29*0.29*0.29+0.29*0.29*0.29*0.29*f)
    
    plt.loglog(f,y1,1+f,y2,2+f,y3)
    plt.xticks(fontsize= 16)
    plt.yticks(fontsize= 16)
    plt.xlabel('Graphene Thickness(ML)', loc ="center", fontsize=18)
    plt.ylabel('Graphene:SiC LEED Intensity Ratio', loc ="center", fontsize=18)
    plt.xlim([0.5, 2.5])
    plt.ylim([0.18, 6])
    plt.xticks([0.1,0.5,0.75,1.0,1.5,2.0,2.5],['0.1','0.5','0.75','1.0','1.5','2.0','2.5'])
    plt.yticks([0.025,0.5,1,2,3,5],['0.025','0.5','1','2','3','5'])
    plt.show()
