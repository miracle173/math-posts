function cNote:
    (FoG)(x) = F(G(x))
# Permutationen von 3 Elementen

         e0 s1 s2 s3 t+ t-
       +------------------
    e0 | e0 s1 s2 s3 t+ t-
    s1 | s1 e0 t+ t- s2 s3
    s2 | s2 t- e0 t+ s3 s1
    s3 | s3 t+ t- e0 s1 s2
    t+ | t+ s3 s1 s2 t- e0
    t- | t- s2 s3 s1 e0 t+
    
    
    e0 ... ()
    s1 ... (23)
    s2 ... (13)
    s3 ... (12)
    t+ ... (123)
    t- ... (132)
    

# Drehung undSpiegelung

         e0 r1 r2 r3 d1 d2 d3 d4
       +------------------------
    e0 | e0 r1 r2 r3 d1 d2 d3 d4
    r1 | r1 r2 r3 e0 d2 d3 d4 d1
    r2 | r2 r3 e0 r1 d3 d4 d1 d2
    r3 | r3 e0 r1 r2 d4 d1 d2 d3
    d1 | d1 d4 d3 d2 e0 r3 r2 r1  
    d2 | d2 d1 d4 d3 r1 e0 r3 r2
    d3 | d3 d2 d1 d4 r2 r1 e0 r3
    d4 | d4 d3 d2 d1 r3 r2 r1 e0
    
## Rotation
    e0 = id
    r1   rotation by pi/2, counterclockwise
    r2 = r1^2
    r3 = r1^3
    

## Reflection
    d1  /
    d2  |
    d3  \
    d4  -
    
