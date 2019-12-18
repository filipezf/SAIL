# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:58:32 2019

@author: u4ft
"""

'''
Language aquisition milestones
------------------------------
basic ontology
references (symbols)
n-ary relationships (social, dynamical)
propositions ( true/false statements)
speech acts (interpersonal)
narratives (multi-statement procedural/declarative)
'''


#generic verbs: v1 v2 v3 v20 ?

def train():
    
    step: e, nn[0,10]
          age [ nn_age]
          gender nn[gender]
          
      a e is N
      a gender is N
      a age is N
      a is live / dead
      
      predict x -> x happens -> positive reinforce


nn[ag1][e] -> predict

x:txt -> transform ->> action

    rule( input x give -> txt: x 'give')  inner txt
    rule( x give -> speak x 'give' )
    
    rule( x give y -> speak x 'give' y )
    rule( x hit y -> speak x 'hit' y )

    rule( input x speak tx -> txt: x 'speak' txt)  inner txt
    rule( x speak txt y -> speak x 'txt' y )


    input : x v y / z v w      txt ->    x v y WHILE z v w
 pick svo1 svo2   txt svo1 before svo2
  pick svo1 svo2   txt svo2 after svo1

     x v y -> setTimeout( txt:  x v y -ed )
     txt x v y will, ->  setTimeout( x v y, 0-5)
     
     txt x v y soon, ->  setTimeout( x v y, 0-5)
      txt x v y later, ->  setTimeout( x v y, 10-20)

     
     input a vb avc avb  a v b 3 times
     input avb    txt avb -ed
     txt  avb

because   <-> txt1 txt2  (causality) % 
IF   <-> txt1 txt2  (causality) %


    input a vo, b vo ->  a AND b vo
          a v o  b v o c vo  -> AND a b c
         a v o a v2 o -> a AND v v2 o
         
         a OR b will vo -> a vo  /// b v o
         
        a NOT vo -> b vo c vo
        x v y ? NOT
        x v y ? YES


    # TODO
    input I/b believe getBelief(b) -> toTxt
    input I/b want getWant(b) -> toTxt
    input I/b can getPossibleAction(b) -> toTxt
    input I/b should getGoodeAction(b) -> toTxt


    
    
# mate



# parenting
u += u_kin


# reciprocal
score[b] += help_hit b


# ingroup / hierarchical
ingroup[b]
rank[x,b]

# outgroup


    s v o ?  
    p ask q
    q answer p

    knowledge, uncertainty?
        (Permit p, q ,txt) ==  (say p, q ( PERM txt)
        
        
    
    (order p, q ,txt   -> q txt 
     order p q txt       NOT q txt  -> p hit q
     
     p say NOT a x y <-> p deny a x y
     
     
    ,(join ,p, g) ==  NOT (IN, p, g ) => (IN ,p ,g)
    ,(separate, p, g)==  (IN, p, g ) => NOT(IN ,p ,g)
    ,(friend ,p, q: p goal => q goal
    ,(enemy ,p, q   p goal => not q goal  
    ,(revenge, b,a) == (hurt,a,b) =>(hurt b a)
    ,(betray a, b)== (believe ,a ,(friend, b,a)) => hurt(b,a)
    ,(learn ,q, txt) ==  (NOT, know ,q ,txt => know ,t, txt) 
    
    
    #,(know ,b, o) == (believe, b,o), o
    
    
   ''' addDef( 'inform  p, q, txt',  '(believe p txt) => (say ,p , q txt)' )
    ,(persuade, p ,q ,txt) == ((NOT (believe ,q, txt); say p ,q, txt), THEN, (believe q txt))

    ,(suggest ,p, txt ==  (say p, (good txt))
    ,(accept p, q txt) == (suggest p q txt, q say do txt)
    ,(decline q, p ,txt)== (suggest p q txt, q say not txt)
    (forbid p, q ,txt )==  (p say  bad q do txt)     
    (lie ,q,p, txt) == (believe p txt);(  say ,q ,not txt) 
    #pretend p txt:  p act if txt
    #suppose p txt  : think p if txt'''
   
    # metaphor
    #emotion
    
    p die
    p will ?
    
    
    
    #spirituality
    
    #? x can svo ,x not visible, x good
    

    # narrative ??
    # philosophy
    # math




def self_play(N):
     for t in range(N):
        #set random goal ?
        env.step()
   





    
    
    
    
