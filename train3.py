# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:58:32 2019

@author: u4ft
"""

'''
_ I you someone them
give hit speak 
birth die
IF AND OR NOT YES TRUE because
want believe
before after WHILE soon later _times 
_ed will
can try should

ask answer order obey
friend enemy child


Language aquisition milestones
------------------------------
basic ontology
references (symbols)
n-ary relationships (social, dynamical)
propositions ( true/false statements)
speech acts (interpersonal)
narratives (multi-statement procedural/declarative)
'''


# before, after
 -> semantic predicto
 
 #event processing



def generate_event():
    s = random
    v = random
    o = random

def train():
    
    # references (symbols)
    for i in range(N):
        
       birth  -> x birth
        
       
       
       
       action( speak s v o)
       
       
       
       
       
       
    x want   
    x believe ctx.set(believe)
    a hit b
    x bel a hit b   
       
    #n-ary relationships (social, dynamical)
    
    
    _ed will
    #propositions ( true/false statements)
    
        #IF AND OR NOT YES TRUE because
        
        
        #IF 
        AND
        OR NOT 
        YES 
        TRUE 
        because
    
    
    #before after WHILE soon later _times
    '?'
    
    can try should
    x before y
    
    
    'say me if blah'
    'say me s = ?'
    
    
    self_play(1000)
   
        
    save(model)    


    #planning
    
    
    #goal
    #wm loop steps
    
    
    
    insert_as_concept( x h c because c h x )
    'x h c because c h x'





    #speak()




    # x if y
    
    # x 'ed
     # x will
     
     #x did p -> because q

    # x join against y


  
    
    
    
    # lying / theory of mind
    
    
    # if I do x then b will do y
    

    #def words

    ( NEC txt)
    ,(O TXT)                           # modal: necessary NEC, permitted, forbiden
    ,(P txt) == (NOT (O (NOT TXT))) 
    
    ,(know ,b, o) == (believe, b,o)
    ,(inform , p, q, txt) == ( (believe p txt) => (say ,p , q txt))
    ,(persuade, p ,q ,txt) == ((NOT (believe ,q, txt); say p ,q, txt), THEN, (believe q txt))
    ,(deny, p, txt) ==  (say ,p ,_ , NOT( txt))
    ,(suggest ,p, txt ==  (say p, (good txt))
    ,(accept p, q txt) == (suggest p q txt, q say do txt)
    ,(decline q, p ,txt)== (suggest p q txt, q say not txt)
    ,(order p, q , txt)    
    ,(order p, q ,txt ==  (p say  bad q not do txt)  # !
    ,(ask p, q, txt) ==  (p want q say txt)           # ?
    ,(answer q, p txt) == (p ask q txt, q say txt)
    (Permit p, q ,txt) ==  (say p, q ( PERM txt)
    (forbid p, q ,txt )==  (p say  bad q do txt)     
    (lie ,q,p, txt) == (believe p txt);(  say ,q ,not txt) 
    #pretend p txt:  p act if txt
    #suppose p txt  : think p if txt
     
    ,(join ,p, g) ==  NOT (IN, p, g ) => (IN ,p ,g)
    ,(separate, p, g)==  (IN, p, g ) => NOT(IN ,p ,g)
    ,(friend ,p, q: p goal => q goal
    ,(enemy ,p, q   p goal => not q goal  
    ,(revenge, b,a) == (hurt,a,b) =>(hurt b a)
    ,(betray a, b)== (believe ,a ,(friend, b,a)) => hurt(b,a)
    ,(learn ,q, txt) ==  (NOT, know ,q ,txt => know ,t, txt) 
    
    
    # metaphor
    #emotion
    #spirituality
    # narrative ??
    #philosophy
    # math


def self_play(N):
     for t in range(N):
        #set random goal ?
        env.step()
   
    
    
    
    #tell a story
    
    
    
''''    
# testing
 2here is b
 
 if a hit b then ?:
marry c
g,h,i join attack c'''
     





    
    
    
    