import random
import copy
import math
import numpy as np
from scipy.special import lambertw
R = 6378137 # the radius of the earth


# function to shifting gps position based on bearing and distance
def shift_gps(gps_start, bearing_mal, d, R):

#Shifting the original gps point to 

#Args:
#    gps_start: the starting point
#    bearing: the direction
#    distance:
    
#Return:
#    gps_end: the end point

    #print('original gps position: lat=%f,log=%f'%(gps_start[0],gps_start[1]))
    gps_start_rad = [math.radians(gps_start[0]), math.radians(gps_start[1])]
    gps_mal_lat = math.asin(math.sin(gps_start_rad[0])*math.cos(d/R) +
                            math.cos(gps_start_rad[0])*math.sin(d/R)*math.cos(bearing_mal))
    gps_mal_log = gps_start_rad[1] + math.atan2(math.sin(bearing_mal)*math.sin(d/R)*math.cos(gps_start_rad[0]),
                                                math.cos(d/R)-math.sin(gps_start_rad[0])*math.sin(gps_mal_lat))
    gps_mal_lat = math.degrees(gps_mal_lat)
    gps_mal_log = math.degrees(gps_mal_log)
    
    gps_new = []
    gps_new.append(gps_mal_lat)
    gps_new.append(gps_mal_log)
    #gps_start[0] = gps_mal_lat
    #gps_start[1] = gps_mal_log
    #print('modified gps position: lat=%f,log=%f'%(gps_new[0],gps_new[1]))
    
    return gps_new


#  You will need to install scipy package in python to use "lambertw" function
def inverseCumulativeGamma(epsilon, z):
    x = (z-1) / math.e
    return - (lambertw(x,k=1) + 1) / epsilon


#  Note epsilon represents privacy level. The smaller epsilon is, the more privacy a user has 
# Input: epsion: the privacy level a user will choose. I used 0.1, 0.01, and 0.001 in my testing
#        pos: original GPS position
# output: pos_new: the new GPS position with Laplace noise added
# [Note: R 
def addLaplaceNoise(epsilon, pos):
    #random number in [0, 2*PI)
    theta = np.random.random() * math.pi * 2
    theta_deg = np.degrees(theta)
    #random variable in [0,1)
    z = np.random.random()
    r = inverseCumulativeGamma(epsilon, z)

    pos_new = shift_gps(pos,theta_deg,r,R)
    #pos_new =  shift_gps(pos,90,300,R)
    #print('adding Laplacian noise to gps position: lat=%f,log=%f'%(pos_new[0],pos_new[1]))
    #return theta_deg,r
    return pos_new