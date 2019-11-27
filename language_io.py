# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:27:01 2019

@author: u4ft
"""


    
words = '''
_ I you someone them move give speak hit
birth die IF AND OR NOT YES TRUE because
TRY can should want believe
before after WHILE soon later _times here there _ed will
very almost like rarely seldom sometimes often always
equal big small positive negative zero one two three smaller bigger
first next last word _list no few some many ALL 
same different that like of 
when what who where why which whom whither how how-much
join separate oppose friend enemy revenge betray
ask answer order obey forbid allow call promise swear bless curse
err forget teach learn just free caring holy loyal respect
mate feed mated mymate mateable isMateage taskmate mate_me
Q X
'''.split()

# Question = ? eXecute = !

namedict = {}
s,v,o,t = range(4)

idx = 1
LEN = 50
vec = {}
ivec = {}
for w in words:
    idx += 1
    globals()[w] = idx
    vec[idx] =  w #np.random.random(LEN)
    
    
PARSER = False

if PARSER:
    import spacy
    
def name2agent(name):
    return namedict[ name.lowercase() ]
    
def txt2op(txt):
   'spacy'
        
    
def op2txt(z):
    z2 = z
    v = vec[z.v]
    if t == '_ed': v = v+'_ed'
    if t == will: v = 'will '+v
    if z.v in [birth ,die]:
         return str(z.s) +' ' + v
    if z.v in [mate, give, hit]:
         return str(z.s) +' '+ v+' '+ str(z.o)
    if z.v in [ believe, want]:  
         return str(z.s) +' '+ v+ ' that '+ op2txt(z.txt)
    if z.v in [speak]  :  
        return str(z.s) +' '+ v + ' to '+ str(z.o)+' that ' + '('+op2txt(z.txt)+')'
    if z.v in [ TRY,can,should]:
        txt2 = copy.copy(z)
        if txt2.s == s:
             txt2.s = ''
        return str(z.s) +' '+ v+  str(s2) + '('+op2txt(txt2)+')' 