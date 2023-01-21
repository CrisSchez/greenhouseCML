import os
os.system("pip install -r requirements.txt")

import pandas as pd
import requests
import numpy as np
import pylab as plt

data = pd.read_csv('sensorConditions.txt', sep=",")
data.columns = ["sensorid", "temperature", "humidity"]
zonedata = pd.read_csv('zoneConditions.txt', sep=",")
zonedata.columns = ["zoneid", "temperature", "humidity"]

# Install the requirements


# Create the directories and upload data

from cmlbootstrap import CMLBootstrap
from IPython.display import Javascript, HTML
import os
import time
import json
import requests
import xml.etree.ElementTree as ET
import datetime
import os
import sys

import os
import sys


from impala.dbapi import connect
from impala.util import as_pandas



from cmlbootstrap import CMLBootstrap
# Set the setup variables needed by CMLBootstrap
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)
IMPALA_HOST='coordinator-se-sandbox-impala-cdw.dw-se-sandboxx-aws.a465-9q4k.cloudera.site'
impala_environment_params = {"IMPALA_HOST":IMPALA_HOST}
storage_environment = cml.create_environment_variable(impala_environment_params)
variables=cml.get_environment_variables()
IMPALA_HOST=variables['IMPALA_HOST']
USERNAME=variables['PROJECT_OWNER']
uservariables=cml.get_user()
USERPASS=uservariables['environment']['WORKLOAD_PASSWORD']
variables
# Connect to Impala using Impyla
# Secure clusters will require additional parameters to connect to Impala.
# Recommended: Specify IMPALA_HOST as an environment variable in your project settings

IMPALA_PORT='443'
#jdbc:impala://coordinator-ClouderaEssencesHue.dw-demo-cloudera-forum-cdp-env.djki-j7ns.cloudera.site:443/default;AuthMech=3;transportMode=http;httpPath=cliservice;ssl=1;UID=cristina.sanchez;PWD=PASSWORD
  
def controlFunc(args):
    sensorid=args["sensorid"]
    temperature=float(args["temperature"])
    humidity=float(args["humidity"])
    try:
      temperatureT=data[data['sensorid']==int(sensorid)]['temperature'].values[0]
      humidityT=data[data['sensorid']==int(sensorid)]['humidity'].values[0]
    except:
      zone=sensorid[0]
      temperatureZ=zonedata[zonedata['zoneid']==int(zone)]['temperature'].values[0]
      humidityZ=zonedata[zonedata['zoneid']==int(zone)]['humidity'].values[0]
            
           
      with open('sensorConditions.txt', 'a') as file:
        file.write('\n'+str(sensorid)+','+str(round(temperatureZ+np.random.uniform(-10,10,1)[0],2))+','+str(humidityZ+round(np.random.uniform(-15,10,1)[0],2)))
                
      data = pd.read_csv('sensorConditions.txt', sep=",")
      data.columns = ["sensorid", "temperature", "humidity"]
      temperatureT=data[data['sensorid']==int(sensorid)]['temperature'].values[0]
      humidityT=data[data['sensorid']==int(sensorid)]['humidity'].values[0]
    
    pointTarget=[(temperatureT,humidityT)]
    point0=[(float(temperature), float(humidity))]
    
    

    checkT=abs(pointTarget[0][0]-point0[0][0])
    checkH=abs(pointTarget[0][1]-point0[0][1])
    
    
    if max(checkT,checkH)>1:
      aviso=1
    else:
      aviso=0
      
    
    return aviso
conn = connect(host=IMPALA_HOST,
               port=IMPALA_PORT,
               auth_mechanism='LDAP',
               user=USERNAME,
               password=USERPASS,
               use_http_transport=True,
               http_path='/cliservice',
               use_ssl=True)
cursor = conn.cursor()

# Execute using SQL

#cursor.execute("drop table if exists lastTS;")
cursor.execute("WITH added_row_number AS (SELECT *, ROW_NUMBER() OVER(PARTITION BY sensorid ORDER BY 'timestamp' DESC) AS row_number FROM greenhouse.greenhouse_sensor) SELECT  * FROM added_row_number WHERE row_number = 1;")
df=as_pandas(cursor)
conn.close()

aviso=[]
for i in range(len(df.index)):
  aviso.append(controlFunc(df.iloc[i]))

df['aviso']=aviso


  
import os, sys
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np

  
from matplotlib.patches import Circle, Wedge, Rectangle

def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points
  
def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation
  

  
def gauge(labels=['LOW','MEDIUM','HIGH','VERY HIGH','EXTREME'], \
          colors='jet_r', arrow=1, title='', fname=False): 
    
    """
    some sanity checks first
    
    """
    
    N = len(labels)
    
    if arrow > N: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N))
 
    
    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors 
    """
    
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))

    """
    begins the plotting
    """
    
    fig, ax = plt.subplots()

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]
    
    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
    
    [ax.add_patch(p) for p in patches]

    
    """
    set the labels (e.g. 'LOW','MEDIUM',...)
    """

    for mid, lab in zip(mid_points, labels): 

        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=14, \
            fontweight='bold', rotation = rot_text(mid))

    """
    set the bottom banner and the title
    """
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)
    
    ax.text(0, -0.05, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=22, fontweight='bold')

    """
    plots the arrow now
    """

    pos = mid_points[min(abs(arrow - N),len(mid_points)-1)]
    
    
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=200)


        
        
 #fin de la función       
       


dfAviso=df[df.aviso==1]      
a=len(df.index)-len(dfAviso.index)
valores=np.linspace(0,len(df.index),6)
import bisect
flecha=bisect.bisect_left(valores, a)

gauge(labels=['CRITICAL','LOW','NOT GOOD','CARE','GOOD'], colors=['#ED1C24','#F18517','#FFCC00','#0063BF','#007A00'], arrow=flecha, title='Sensores defectuosos ='+str(len(dfAviso.index))) 
plt.savefig('numSensors.png')
print(dfAviso['sensorid'].values)
    

  
def mincost(args):
    sensorid=args["sensorid"]
    temperature=float(args["temperature"])
    humidity=float(args["humidity"])
    temperatureT=data[data['sensorid']==int(sensorid)]['temperature'].values[0]
    humidityT=data[data['sensorid']==int(sensorid)]['humidity'].values[0]
    pointTarget=[(temperatureT,humidityT)]
    point0=[(float(temperature), float(humidity))]
    
    
    url = "https://api.tomorrow.io/v4/timelines"

    querystring = {
    "location":"40,-4",
    "fields":["temperature", "humidity"],
    "units":"metric",
    "timesteps":"1d",
    "apikey":"AnYikfGuNWW3A8OIqfLHG3L5uvhIUnNn"}

    response = requests.request("GET", url, params=querystring)
    t = response.json()['data']['timelines'][0]['intervals'][0]['values']
    pointExt=[(t['temperature'],t['humidity'])]
    

    tipoT=[]
    tipoH=[]
    if point0[0][0]>pointExt[0][0]:
        if pointTarget[0][0]>point0[0][0]:
            tipoT=0
        else:
            if pointTarget[0][0]>pointExt[0][0]:
                tipoT=2
            else:
                tipoT=1
    else: 
        if pointTarget[0][0]>pointExt[0][0]:
            tipoT=1
        else:
            if pointTarget[0][0]>point0[0][0]:
                tipoT=2
            else:
                tipoT=0



    if point0[0][0]>pointExt[0][0]:
        if pointTarget[0][0]>point0[0][0]:
            tipoH=0
        else:
            if pointTarget[0][0]>pointExt[0][0]:
                tipoH=2
            else:
                tipoH=1
    else: 
        if pointTarget[0][0]>pointExt[0][0]:
            tipoH=1
        else:
            if pointTarget[0][0]>point0[0][0]:
                tipoH=2
            else:
                tipoH=0
                
                

    nh=4
    pointH=[]
    for i in range(nh):
        pointH.append(pointTarget[0][1]-(pointTarget[0][1]-point0[0][1])*i/(nh-1))
    nt=4 
    pointT=[]
    for i in range(nt):
        pointT.append(pointTarget[0][0]-(pointTarget[0][0]-point0[0][0])*i/(nt-1))
        
    pointsGrid=[]
    for i in range(nh):
        for j in range(nt):
            pointsGrid.append((pointH[i],pointT[j]))
            
    goal=0
    # create matrix x*y
    RT = np.matrix(np.zeros(shape=(nh*nt, nh*nt)))
    RH = np.matrix(np.zeros(shape=(nh*nt, nh*nt)))
    R = np.matrix(np.zeros(shape=(nh*nt, nh*nt)))
    
    if tipoT==0:
        for i in range(len(pointsGrid)):
            for j in range(len(pointsGrid)):
                RT[i,j]=pow((pointsGrid[i][1]+pointsGrid[j][1]),2)
    if tipoT==1:
        for i in range(len(pointsGrid)):
            for j in range(len(pointsGrid)):
                RT[i,j]=pow((pointsGrid[i][1]-pointExt[0][0]),2)+pow((pointsGrid[j][1]-pointExt[0][0]),2)

    if tipoT==2:
        for i in range(len(pointsGrid)):
            for j in range(len(pointsGrid)):
                RT[i,j]=pow((pointsGrid[i][1]-pointsGrid[j][1]),2)

    if tipoH==0:
        for i in range(len(pointsGrid)):
            for j in range(len(pointsGrid)):
                RH[i,j]=pow((pointsGrid[i][0]+pointsGrid[j][0]),2)
    if tipoH==1:
        for i in range(len(pointsGrid)):
            for j in range(len(pointsGrid)):
                RH[i,j]=pow((pointsGrid[i][0]-pointExt[0][1]),2)+pow((pointsGrid[j][0]-pointExt[0][1]),2)

    if tipoH==2:
        for i in range(len(pointsGrid)):
            for j in range(len(pointsGrid)):
                RH[i,j]=pow((pointsGrid[i][0]-pointsGrid[j][0]),2)

    for i in range(nt*nh):
        for j in range(nt*nh):
            if np.sqrt(RT[i,j]*RH[i,j])==0:
                R[i,j]=99999
            else:
                R[i,j]=1/np.sqrt(RT[i,j]+RH[i,j])

        R[i,i]=-1
        R[goal,goal]=99999

        
        Q = np.matrix(np.zeros([nh*nt,nh*nt]))

    # learning parameter
    gamma = 0.8

    initial_state = len(pointsGrid)-1

    def available_actions(state):
        current_state_row = R[state,]
        av_act = np.where(current_state_row >= 0)[1]
        return av_act

    available_act = available_actions(initial_state) 

    def sample_next_action(available_actions_range):
        next_action = int(np.random.choice(available_act,1))
        return next_action

    action = sample_next_action(available_act)

    def update(current_state, action, gamma):

        max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size = 1))
        else:
            max_index = int(max_index)
        max_value = Q[action, max_index]

        Q[current_state, action] = R[current_state, action] + gamma * max_value
        #print('max_value', R[current_state, action] + gamma * max_value)

        if (np.max(Q) > 0):
            return(np.sum(Q/np.max(Q)*100))
        else:
            return (0)

    update(initial_state, action, gamma)
    
    # Training
    scores = []
    for i in range(1000):
        current_state = np.random.randint(0, int(Q.shape[0]))
        available_act = available_actions(current_state)
        action = sample_next_action(available_act)
        score = update(current_state,action,gamma)
        scores.append(score)
        #print ('Score:', str(score))

    #print("Trained Q matrix:")
    #print(Q/np.max(Q)*100)

    # Testing
    current_state = len(pointsGrid)-1
    steps = [current_state]
    count=0
    while (current_state != 0):

        next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

        if next_step_index.shape[0] > 1:
            next_step_index = int(np.random.choice(next_step_index, size = 1))
        else:
            next_step_index = int(next_step_index)
        if next_step_index>current_state:
            next_step_index=0
        steps.append(next_step_index)

        current_state = next_step_index
  

    print("Most efficient path:")
    print(steps)
    output=[]
    for i in range(len(steps)):
        output.append(pointsGrid[steps[i]])
    return output

    #plt.plot(scores)
    #plt.show()

        
