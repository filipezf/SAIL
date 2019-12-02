# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:24:52 2019

@author: u4ft
"""

import env
import random
import numpy as np
import random
import os 
import copy
import itertools
import base_agent as ba
import torch


pop = env.pop

def coin(p):
    return random.random() < p
    
'''nn_before = NNSupervised()
nn_after = NNSupervised()
nn_bdi = NNSupervised()
nn_can = NNSupervised()'''
        

# some common-sense 
       #x speak y -> z believe y
    
# model of above
    #nn[my_attention] = 

# play dream imagine
    
Z = ba.Z    
cnt = 0
    
class Agent(ba.Agent):
        
    #def  __init__(self,m=None,f=None, gender=None):
    
        
    def __str__(self):
        return 'a_' + str(self.id)                        
      
    def step(self):  
        self.step0()
        
        actions = self.actions = []
        
        
      
        if coin(0.33):
            j = self.speakDummy()
            if j is not None:
                actions.append( j )
            
            
        for inp in self.log:
                   # a said to b that c loves d
                   # s=a v=say t=-ed, o=b, txt=(c loves d)
                   # [ a b c d ab ac ... VEC998 ]
           
            if coin(0.33):
                self.hearDummy( inp )
            vinp = self.toVec( inp)    

       
            # model-free
            vout = self.NN ( torch.from_numpy(vinp).float() )
            
            act = self.toZ( vout )
            if act.confidence > 0.5:
                actions.append( act )
            
            #model based / plan        
            
            
        if coin(0.3):            
        #if max( [act.confidence for act in actions]) < 0.5:
        
            options = []
            for a  in self.sampleActions(3):                
                c = self.getConsequences(a )
                u = self.Q(c)
                options.append( (a ,u) )
                
            options.sort(key = lambda x :x[1].item()) 
                
            j = options[0][1]
            actions.append( self.toZ(j) )         
        
        
        #self.updateRL(a,c,u)        
        #self.updateSupervisedNetworks(a,c)
        
        self.log = []
     
             
    def input1(self, foo):
        if foo.v == ba.mated:
            self.mated = True
            return       
        #self.em_write(foo)
        self.log.append( foo )
        
        
    def speakDummy(self):
        i = random.randint(-1,8)
        
        if i==-1:
            ev = self.getPastEvent(-1)
            if ev is None: return None
            ev.t = None
        
        if i==0:
            ev = self.getPastEvent()
            if ev is None: return None
            ev.t = ba._ed
        
        if i==1:
            ev = self.predictFutureEvent()
            if ev is None: return None
            ev.t = ba.will
        
        if i==2: 
            a = self.generateRandomEvent()
            b = self.predictFuture(a)     
            
            ev = Z(s = a, v= ba.IF, txt=b)
            
        if i==3: 
            a = self.getPastEvent()
            if a is None: return None
            b = self.getCause(a)              
            ev= Z(s = a, v= ba.because, txt=b)

        if i==4:
            ag = random.choice(pop)
            ev = self.getPossibleAction(ag)
            ev.m = ba.can

        if i==5:
            a = random.choice(pop)
            ev = self.getBelief(a)
            if ev is None: return None
            ev= Z(s = a, v= ba.believe, txt=ev)            
            
        if i==6:
            a = random.choice(pop)
            ev = self.getGoal(a)
            if ev is None: return None
            ev = Z(s = a, v= ba.want, txt=ev)    
        
        if i==7:
            ev = self.getQuestion()
            ev.mod = ba.QU # ?
        
        if i==8:
            ev = self.getCmd()
            ev.mod = ba.X # !
              

        return ev
        
    def hearDummy(self, ev):
        return 
        v,t,m,mod = ev.v, ev.t, ev.m, ev.mod
        if t== None:    self.addEpMem(ev)  
        elif t == _ed:    self.addEpMemPast(ev)
        elif t == will:  self. addPssibleFuture(ev)
        elif v == IF:    self.setPredictModel(a,b)
        elif v==because:   self.setPredictModel(a,b)
        elif m==can:      self.setPossibleAction(a, txt)
        elif v==believe:  self.writeBelief(a, txt)
        elif v==want:     self.writeGoal(a, txt)        
        elif mod == '?':  self.answer(ev)        
        elif mod == '!': self.setGoal(ev)
