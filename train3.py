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
    s = random.choice(pop)
    v = random.choice( VERBS)
    o = random.choice(pop)

def train():
    
    # references (symbols)
    for i in range(N):
        
       s1 = random.choice(pop)
       v1 = random.choice( VERBS)
       o1 = random.choice(pop)      
       a.input1( s=s1, v=v1, o=o1 )       
       a.action( speak s v o)
       supervised()
       
       y give x, good x
       x mate good
       'is good'
       parse ?
       
       
       # concepts  too advnced for the puny human mind to grasp
       
     a.input1( s=s1, v=v1, o=o1 )    
    setTimeout( 5/N ,  s t=_ed v=v1, o=o1  )
       
    inputEvent( 'x want', ''  ) 
    x believe ctx.set(believe)
    a hit b
    x bel a hit b   
       
    #n-ary relationships (social, dynamical)
    
    
    a.input1( s=s1, v=v1, o=o1 )    
    setTimeout( 5/N ,  s t=_ed v=v1, o=o1  )
        
    a.input1( s=s1, v=v1, o=o1 t=will )
    setTimeout(a.input1( s=s1, v=v1, o=o1 )
     
    #propositions ( true/false statements)
    
        #IF AND OR NOT YES TRUE because
        
        # situations
        
        # and
        'a and b hit c -> a hit c, b hit c'
        'a hit and give c -> a hit c and give c'
        
         # or
        'a and b hit c -> a hit c, b hit c'
        'a hit and give c -> a hit c and give c'
         
        #IF 
        'if svo, o= svo'
        
     
        'x hit c'
        'x NOT hit y'
        
        #YES 
        'x h y'
        'x h y ?'
        'yes'
        
        #TRUE 
        
        #because
        'a speak b because a mother b'
        'a mother b' -> 'a birth b'
    
    
    #before 
    svo before svo
    'x hit y before c give y'
    after
    
    #after 
    
    
    
    #WHILE 
    '?'
    
    #can 
    
    
    #try 
    
    
    #should
    
    
      
    self_play(1000)
   
        
    save(model)    


    #planning
    #goal
    #wm loop steps
    
    
 
    # x if y
    
    # x 'ed
     # x will
     
     #x did p -> because q

    # x join against y


  
    
    
    
    # lying / theory of mind
    
    
    # if I do x then b will do y
    

    #def words

    #( NEC txt)
    #,(O TXT)                           # modal: necessary NEC, permitted, forbiden
    #,(P txt) == (NOT (O (NOT TXT))) 
    
    
    addDef(    ) # do you know/believe what they said of you?
    
    
    #,(know ,b, o) == (believe, b,o)
    addDef( 'inform  p, q, txt',  '(believe p txt) => (say ,p , q txt)' )
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
   





    
    
    
    
