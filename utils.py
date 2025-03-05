import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from random import shuffle
# plt.style.use('ggplot')


def function_mapping(beta,c_const,beta_l,c_const_l):
#     %this function determine which type of functional form, types of concave
#     % and the critical point of size according to the parameters of the market
    
#     % beta: power law exponent of net income
#     % c_const: power law intercept of net income
#     % beta_l: power law exponent of liability
#     % c_const_l: power law intercept of liability
#     % ff: functional form
#     % concave: 1 for A''(t)>0, 0 for A''(t)<0, and -1 for A(t) is exponential
    c_const = np.e**c_const
    c_const_l = np.e**c_const_l
    onebeta = 1 - beta
    oneminus = onebeta * c_const_l * beta_l / (beta_l - beta)
    betac = onebeta * c_const
    betalminus = beta_l - 1

    def f1(x):
        return pow(x, onebeta) * (1 - oneminus * pow(x, betalminus)) / betac
    def f2(x):
        return ((np.log(x) - pow(x, betalminus) * beta_l * c_const_l / betalminus)) / c_const
    def f3(x):
        return np.log(x) * (1 - c_const_l) / (c_const)
    def f4(x):
        return pow(x, onebeta) * (1-oneminus * pow(x,betalminus)) / betac
    def f5(x):
        return pow(x, onebeta) / betac - beta * c_const_l * np.log(x) / c_const
    ff=f1
    if abs(beta-1) < 0.01:
        ff = f2
        if abs(beta_l-1) < 0.01:
            ff = f3
    if beta > 1 or beta > beta_l:
        ff = f4
    if abs(beta-beta_l) < 0.01:
        ff = f5
#   if beta_l<>1, then calculate concave
    if abs(beta_l-1) > 0.01 and c_const_l > 0 and beta_l > 0:
#         solving astar: dA/dt=cA^b/(1-cl bl A^(bl-1))=0
        astar = np.exp((np.log(c_const_l) + np.log(beta_l)) / (1 - beta_l))
        # solving concave by d^2A/dt^2>0
        concave = beta * pow(astar,(1 - beta_l)) > c_const_l * beta_l * (-beta_l + beta + 1)
    else:
        astar=np.nan
        concave=-1
    return ff,concave,astar



def err(true, pred):
    true_new = []
    pred_new = []
    err = []
    for i,item in enumerate(true):
        if item!=0:
            true_new.append(true[i])
            pred_new.append(pred[i])
    err = np.array(true_new) - np.array(pred_new)
    err = np.sign(np.mean(err)) * np.mean(abs(err))
    return err

def laplacy_normolize(error):
    u = np.mean(error)
    b = sum([abs(i-u) for i in error]) / len(error)
    return u,b

def getLog(x):
    x = np.array(x)
    return np.sign(x)*(np.log(abs(x)+1))
def getExp(x):
    x = np.array(x)
    return np.sign(x)*(np.exp(x)-1)

def getLog10(x):
    x = np.array(x)
    return np.sign(x)*(np.log10(abs(x)+1))
def getPow10(x):
    x = np.array(x)
    return np.sign(x)*(pow(10,abs(x))-1)

def get_predict_and_err(assets,times,beta,c_const,beta_l,c_const_l,size_range,time_range):
    assets = np.array(assets)
    times = np.array(times)
    ff,concave,astar=function_mapping(beta,c_const,beta_l,c_const_l)
    c_const = np.e**c_const
    c_const_l = np.e**c_const_l
    'select bifurcation'
    plotx=[]
    ploty=[]
    if np.isnan(astar):
        if abs(beta_l-1) < 0.01:
            # beta_l -> 1
            if abs(beta-1) < 0.01:
                # beta_i -> 1
#                 predict = c_const * times / (np.log(10) * (1-c_const_l));
                predict = np.exp(times / (1-c_const_l))
            else:
#                 predict=(np.log((1-beta)*times/(1-c_const_l))+np.log(c_const))/(1-beta);
                predict = pow(c_const * times * (1-beta) / (1-c_const_l), 1 / (1-beta))
            err = np.log(assets)-np.log(predict)
            return err,predict
        else:
            yrange = np.logspace(np.log10(size_range[0]),np.log10(size_range[-1]),1000)
            xrange = np.real(ff(yrange))
            err = get_errors(assets,times,xrange,yrange,plotx,ploty,concave)
            return err,np.exp(predict)
    if astar > 0:
        if concave:
#           if the bifurcation is toward left
#           padding right, that is adding prediction points for constants
            tt = max(np.append(time_range,ff(astar)+len(times)))
            plotx = np.linspace(ff(astar),(tt+1),round(tt+1 - ff(astar)))
            ploty = np.ones([1,len(plotx)])*astar
        else:
#           if the bifurcation is toward right
            ran = ff(astar) - min (time_range) + 2
#           padding left points for constants
            plotx = [ff(astar) - ran, ff(astar)]
            ploty = np.ones(len(plotx)) * astar
#   to determine which the branch of the bifurcation, and calculate errors
    if assets[0] > astar:
#         %upper bif
        yrange = np.logspace(np.log10(astar),np.log10(size_range[-1]),10000)
        xrange = ff(yrange)
        err,predict = get_errors(assets,times,xrange,yrange,plotx,ploty,concave)
        return err,np.exp(predict)
    else:
#         %lower bif
        yrange = np.logspace(np.log10(size_range[0]),np.log10(astar),10000)
        xrange = ff(yrange)
        err,predict = get_errors(assets,times,xrange,yrange,plotx,ploty,concave)
        return err,np.exp(predict)


def get_errors(assets, times, xrange, yrange, plotx, ploty, concave):
    #  % calculate the absolute values under given x,y ranges and plotx, ploty
    #     % (theoretical prediction)
    #     % xrange: considered x range
    #     % yrange: considered y range
    #     % plotx: padding values on x
    #     % ploty: padding values on y
        
    #     % invert xy range if they are in inverse orders
    if xrange[-1] < xrange[0]:
        xrange = xrange[::-1] #ex: transe [1,2,3,4] to [4,3,2,1]
        yrange = yrange[::-1]
    #    % supplementing x,y ranges according to plotx,ploty.
    if concave:
        xrange= np.append(xrange,plotx)
        yrange = np.append(yrange,ploty)
    #         if len(plotx)>1:
    #             np.append(xrange, plotx[-1]+1)
    #             np.append(yrange, ploty[-1])
    else:
        '右开口，在前面插入临界点'
        np.insert(xrange,0,plotx)#[plotx, xrange];
        np.insert(yrange,0,ploty)
    err = np.zeros(len(times))
    pred = np.zeros(len(times))
    #     % searching all possible times to find the optima(minimize errors)
    #     % and calculate errors
    for i in range(len(times)):
        bools = times[i]>=xrange
#         % finding zero (cross) points
        x1 = np.where(bools)[0]
        x2 = np.where(~bools)[0]
        if sum(bools)<1 or sum(~bools)<1:
#             % if no such points, then err will be set to be asset itself
            err[i] = 0#np.log(assets[i]);
#             err[i]=np.log(assets[i]);
            print(times[i])
        else:
#             % this fractional point should be considered because there may
#             % exist many points for equating (times(i)=xrange)
#             % finding the last times(i)>xrange point
            x1 = x1[-1]
#             % finding the first times(i)<xrange point
            x2 = x2[0]
#             % calculating their proportion on times
            a = (times[i] - xrange[x1]) / (xrange[x2] - xrange[x1])
#             % according to the ratio lambda to calculate y
            y = np.log(yrange[x1]) * a + (1 - a) * np.log(yrange[x2])
            pred[i] = y
            err[i] =np.log(assets[i]) - y
    return err,pred



def loglogbind(aa, bb, bins=30):
    xx = []
    yy = []
    yy_var = []
    aa = np.array(aa)
    bb = np.array(bb)
    binx = np.histogram(getLog10(aa),bins)
    binx = binx[1]
    for i in range(len(binx)-1):
        a = aa[np.where((aa >= getPow10(binx[i])) & (aa < getPow10(binx[i+1])))]
        if len(a) > 0:
            b = bb[np.where((aa >= getPow10(binx[i]))  & (aa < getPow10(binx[i+1])))]
#             xx.append(getPow10(binx[i]))
            xx.append(np.mean(getLog10(a)))
            yy.append(np.mean(getLog10(b)))
            yy_var.append(np.std(getLog10(b)))

    yy = np.array(yy)
    xx = np.array(xx)
    yy_var = np.array(yy_var)
    return xx,yy,yy_var


def logxbind(aa, bb, bins=30):
    xx = []
    yy = []
    yy_var = []
    aa = np.array(aa)
    binx = np.histogram(getLog10(aa),bins)
    binx = binx[1]
    for i in range(len(binx)-1):
        a = aa[np.where((aa >= getPow10(binx[i])) & (aa < getPow10(binx[i+1])))]
        b = bb[np.where((aa >= getPow10(binx[i])) & (aa < getPow10(binx[i+1])))]
        xx.append(np.mean(getLog10(a)))
        yy.append(np.mean(b))
        yy_var.append(np.std(b))

    yy = np.array(yy)
    xx = np.array(xx)
    yy_var = np.array(yy_var)
    return xx,yy,yy_var

def logybind(aa, bb, bins=30):
    xx = []
    yy = []
    yy_var = []
    aa = np.array(aa)
    binx = np.histogram(aa,bins)
    binx = binx[1]
    for i in range(len(binx)-1):
        a = aa[np.where((aa >= binx[i]) & (aa < binx[i+1]))]
        b = bb[np.where((aa >= binx[i]) & (aa < binx[i+1]))]
        xx.append(np.mean(a))
        yy.append(np.mean(getLog10(b)))
        yy_var.append(np.std(getLog10(b)))

    yy = np.array(yy)
    xx = np.array(xx)
    yy_var = np.array(yy_var)
    return xx,yy,yy_var
def bind(aa,bb,bins=30):
    xx = []
    yy = []
    yy_var = []
    aa = np.array(aa)
    bb = np.array(bb)
    binx = np.histogram(aa,bins)
    binx = binx[1]
    
    for i in range(len(binx)-1):
        a = aa[np.where((aa >= binx[i]) & (aa < binx[i+1]))]
        if len(a) > 0:
            b = bb[np.where((aa >= binx[i])  & (aa < binx[i+1]))]
            xx.append(np.mean(a))
            yy.append(np.mean(b))
            yy_var.append(np.std(b))
    return xx,yy,yy_var

def set_plot_basicinf(ylabel,xlabel,fontsize=14):
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    
    

