import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import keras
from keras import models
from keras import layers
#from keras.callbacks import TensorBoard
#%% Data preparation
np.random.seed(10235)
g=9.8
theta = np.random.rand(1100,1)*(np.pi/2)
Vint =  np.random.rand(1100,1)*20
'''
#show distribution
fig = plt.figure(figsize=[10.,8.])
thetaplot=sns.distplot(theta,bins=10)
thetaplot.set_xlabel(r'$\theta$(rad)')
plt.show()
fig = plt.figure(figsize=[10.,8.])
thetaplot=sns.distplot(Vint,bins=10)
thetaplot.set_xlabel('V0(m/s)')
plt.show()
'''
Rmax = 2*Vint**2*np.cos(theta)*np.sin(theta)/g
Hmax = (Vint*np.sin(theta))**2/2/g
#Vint = (Vint-np.mean(Vint))/np.std(Vint) #normalize
#theta = (theta-np.mean(theta))/np.std(theta) #normalize
train_x=np.concatenate([Vint,theta],axis=1)[0:1000]
train_Rmax,train_Hmax=Rmax[0:1000],Hmax[0:1000]
test_x =np.concatenate([Vint,theta],axis=1)[1000:1100]
test_Rmax,test_Hmax=Rmax[1000:1100],Hmax[1000:1100]
#%% Model build
def Model_build():
    act_ = 'relu'
    opti_ = 'adam'
    loss_ = 'mse'
    metri_ = ['mae']
    model = models.Sequential()   
    model.add(layers.Dense(units=256,activation=act_,input_shape=(train_x.shape[1],))) 
    model.add(layers.Dense(units=256,activation=act_))
    model.add(layers.Dense(units=1))
    model.compile(optimizer= opti_,loss=loss_,metrics=metri_)
#    model.summary()
    return model
def smooth_curve(points,factor=0.9):
    smoothed_points= []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
#%% cross validation R
'''
k=8 
val_samples = len(train_x)//k
no_epochs = 500
no_batch = 64
all_mae_histories=[]
all_loss_histories=[]
for i in range(k):
    print('processing #',i)
    val_x = train_x[i*val_samples: (i+1)*val_samples]
    val_Rmax = train_Rmax[i*val_samples: (i+1)*val_samples]
    k_fold_x = np.concatenate([train_x[:i*val_samples],train_x[(i+1)*val_samples:]],axis=0)
    k_fold_Rmax = np.concatenate([train_Rmax[:i*val_samples],train_Rmax[(i+1)*val_samples:]],axis=0)
    Rmodel = Model_build()
    history=Rmodel.fit(k_fold_x, k_fold_Rmax,epochs=no_epochs,validation_data=(val_x,val_Rmax)
                      ,batch_size=no_batch,verbose=0)
    mae_history = history.history['val_mae']
    loss_history = history.history['val_loss']
    all_mae_histories.append(mae_history)  
    all_loss_histories.append(loss_history)


#%% draw validation R
average_mae_histories = np.mean(all_mae_histories,axis=0)
smooth_mae_history = smooth_curve(average_mae_histories[150:])
fig = plt.figure(figsize=[10.,8.])
plt.plot(range(1,len(average_mae_histories)+1),average_mae_histories)
plt.title('R mae all')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.xticks(range(0,501,50))
plt.show()
plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.title('R mae 150: with smooth')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.xticks(range(0,450,50))
plt.show()
average_loss_histories = np.mean(all_loss_histories,axis=0)
plt.plot(range(1,len(average_loss_histories)+1),average_loss_histories)
plt.title('R mse all')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.xticks(range(0,501,50))
plt.show()
smooth_loss_history = smooth_curve(average_loss_histories[150:])
plt.plot(range(1,len(smooth_loss_history)+1),smooth_loss_history)
plt.title('R mse 150: with smooth')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.xticks(range(0,450,50))
plt.show()
#%% cross validation H
k=8 
val_samples = len(train_x)//k
no_epochs = 500
no_batch = 64
all_mae_histories=[]
for i in range(k):
    print('processing #',i)
    val_x = train_x[i*val_samples: (i+1)*val_samples]
    val_Hmax = train_Hmax[i*val_samples: (i+1)*val_samples]
    k_fold_x = np.concatenate([train_x[:i*val_samples],train_x[(i+1)*val_samples:]],axis=0)
    k_fold_Hmax = np.concatenate([train_Hmax[:i*val_samples],train_Hmax[(i+1)*val_samples:]],axis=0)
    Hmodel = Model_build()
    history=Rmodel.fit(k_fold_x, k_fold_Hmax,epochs=no_epochs,validation_data=(val_x,val_Hmax)
                      ,batch_size=no_batch,verbose=0)
    mae_history = history.history['val_mae']
    loss_history = history.history['val_loss']
    all_mae_histories.append(mae_history)  
    all_loss_histories.append(loss_history)

#%% draw validation H
average_mae_histories = np.mean(all_mae_histories,axis=0)
smooth_mae_history = smooth_curve(average_mae_histories[150:])
plt.plot(range(1,len(average_mae_histories)+1),average_mae_histories)
plt.title('H mae all')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.xticks(range(0,501,50))
plt.show()
plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.title('H mae 150: with smooth')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.xticks(range(0,450,50))
plt.show()
average_loss_histories = np.mean(all_loss_histories,axis=0)
plt.plot(range(1,len(average_loss_histories)+1),average_loss_histories)
plt.title('H mse all')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.xticks(range(0,501,50))
plt.show()
smooth_loss_history = smooth_curve(average_loss_histories[150:])
plt.plot(range(1,len(smooth_loss_history)+1),smooth_loss_history)
plt.title('H mse 150: with smooth')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.xticks(range(0,450,50))
plt.show()
'''
#%% training model
def Model_build():
    act_ = 'relu'
    opti_ = 'adam'
    loss_ = 'mse'
    metri_ = ['mae']
    model = models.Sequential()   
    model.add(layers.Dense(units=256,activation=act_,input_shape=(train_x.shape[1],))) 
    model.add(layers.Dense(units=256,activation=act_))
    model.add(layers.Dense(units=1))
    model.compile(optimizer= opti_,loss=loss_,metrics=metri_)
    model.summary()
    #tensorboard --logdir=./graphs --host=127.0.0.1
    return model
callbacks = [keras.callbacks.TensorBoard(log_dir='./graphs')]
no_epochs = 500
no_batch = 64
Rmodel=Model_build()
historyR=Rmodel.fit(train_x, train_Rmax,epochs=no_epochs,batch_size=no_batch
                    ,verbose=0,validation_split=0.2,callbacks=callbacks)
maeR_history = historyR.history['val_mae']
mseR_history = historyR.history['val_loss']
Rmodel.save('Rmax.h5')
Hmodel=Model_build()
historyH=Hmodel.fit(train_x, train_Hmax,epochs=no_epochs,batch_size=no_batch
                    ,verbose=0,validation_split=0.2,callbacks=callbacks)
maeH_history = historyH.history['val_mae']
mseH_history = historyH.history['val_loss']
Hmodel.save('Hmax.h5')
del Rmodel,Hmodel
Rmodel = models.load_model('Rmax.h5')
Hmodel = models.load_model('Hmax.h5')
test_yR = Rmodel.predict(test_x)
test_yH = Hmodel.predict(test_x)
fig = plt.figure(figsize=[10.,8.])
plt.plot(range(1,len(maeR_history)+1),maeR_history)
plt.title('R mae all')
print('maeR is',maeR_history[-1])
plt.xlabel('epochs')
plt.ylabel('mae')
plt.xticks(range(0,501,50))
plt.show()
plt.plot(range(1,len(maeH_history)+1),maeH_history)
plt.title('H mae all')
print('maeH is',maeH_history[-1])
plt.xlabel('epochs')
plt.ylabel('mae')
plt.xticks(range(0,501,50))
plt.show()
#answer=model.predict(test_x)
#%%
Vint,theta =np.meshgrid(np.arange(0,20.1,20/50), np.arange(0,np.pi/2+0.000001,np.pi/100))
pre_x = np.array([Vint.flatten(),theta.flatten()]).T
pre_Rmax = Rmodel.predict(pre_x)
pre_Hmax = Hmodel.predict(pre_x)
pre_y = np.concatenate([pre_Rmax,pre_Hmax],axis=1)
#%% draw
fig = plt.figure(figsize=[32.,18.])
Rmaxplot = fig.gca(projection='3d')
x,y=np.arange(0,20.1,20/50),np.arange(0,np.pi/2+0.000001,np.pi/100)
X, Y = np.meshgrid(x, y)
Z = pre_y[:,0].reshape([51,51])
Rmaxplot.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.5, cmap=cm.coolwarm)
Rplot=Rmaxplot.contourf(X, Y, Z, zdir='z',offset=-1, alpha=0.9, cmap=cm.coolwarm)
Rmaxplot.set_xlabel('Vint')
Rmaxplot.set_ylabel('rad')
Rmaxplot.set_zlabel('Rmax')
Rmaxplot.set_yticks(np.arange(0,np.pi/2+0.1,np.pi/8))
fig.colorbar(Rplot, shrink=0.5, aspect=5)
Rmaxplot.view_init(elev=40,azim=300)
plt.close
fig = plt.figure(figsize=[32.,18.])
Hmaxplot = fig.gca(projection='3d')
Z = pre_y[:,1].reshape(X.shape)
Hmaxplot.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.5, cmap=cm.coolwarm)
Hplot=Hmaxplot.contourf(X, Y, Z, zdir='z',offset=-1, alpha=0.9, cmap=cm.coolwarm)
Hmaxplot.set_xlabel('Vint')
Hmaxplot.set_ylabel('rad')
Hmaxplot.set_zlabel('Hmax')
Hmaxplot.set_yticks(np.arange(0,np.pi/2+0.1,np.pi/8))
fig.colorbar(Hplot, shrink=0.5, aspect=5)
Hmaxplot.view_init(elev=40,azim=300)
plt.close
