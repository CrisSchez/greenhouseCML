import pandas as pd
import requests
import numpy as np
import pylab as plt

data = pd.read_csv('sensorConditions.txt', sep=",")
data.columns = ["sensorid", "temperature", "humidity"]
zonedata = pd.read_csv('zoneConditions.txt', sep=",")
zonedata.columns = ["zoneid", "temperature", "humidity"]


def mincost(args):
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

        
