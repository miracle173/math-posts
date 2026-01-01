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

    Anmerkung: d3 ist die Matrix-Transposition


ist p eine Permutation, dann bezeichne Rp eine Band-Permutation und Cp eine Stackpermutation, wobei p angibt, wie die Bänder 0,1,2 bzw. die Stacks 0,1,2 Permutiert werden

Falsch:
Die Matrix A

    ( a_00, a_01, a_02, ..., a_08 )
    ( a_10, a_11, a_12 ,..., a_18 )
    (               ...           )
    ( a_80, a_81, a_82, ..., a_88 )

wird durch Rp(A) transformiert zu

    ( a_p(0)0, a_p(0)1, a_p(0)2, ..., a_p(0)8 )
    ( a_p(1)0, a_p(1)1, a_p(1)2, ..., a_p(1)8 )
    (             ...                         )
    ( a_p(8)0, a_p(8)1, a_p(8)2, ..., a_p(8)8 )

und durch Cp(A) zu

    ( a_0p(0), a_0p(1), a_0p(2), ..., a_0p(8) )
    ( a_1p(0), a_1p(1), a_1p(2), ..., a_1p(8) )
    (                 ...                     )
    ( a_8p(0), a_8p(1), a_0p(2), ..., a_8p(8) )

ende falsch

ist s die Spiegelung d3, also die Matrixtranposition, dann ist

    s o R_p = C_p o s
    s o C_p = R_p o s
    s o r_{i,p} = c_{i,p} o s
    s o c_{i,p} = r_{i,p} o s

    r_{i, p1} o r_{i, p2} = r_{i, p1 o p2}
    r_{i, p1} o r_{j, p2} = r_{j, p1} o r_{i, p2}, falls i != j
    
    r_p1 o r_p2 = r_{p1 o p2}
    r_p1 o r_p2 = r_{p1 o p2}
    r_p1 o r_p2 = r_{p1 o p2}

           |s      |r_{i,q}        |c_{i,q}        |R_q                |C_q
    -------+-------+---------------+---------------+-------------------+-----------------
    s      |1      |c_{i,q}*s      |r_{i,q}*s      |C_q*s              |R_q*s
    r_{i,p}|s*c_i,p|r_{i,p*q}      |c_{i,q}*r_{i,p}|R_q*r_{q^(-1)(i),p}|C_q*r_{i,p
    r_{j,p}|~      |r_{i,q}*r_{j,p}|c_{i,q}*r_{j,p}|~                  |~
    c_{i,p}|s*r_i,p|r_{i,q}*c_{i,p}|c_{i,p*q}      |R_q*c_{i,p}        |C_q*c_{q^(-1)(i),p}
    c_{j,p}|~      |r_{i,q}*c_{j,p}|c_{i,q}*c_{j,p}|~                  |~
    R_p    |s*C_p  |r_{p(i),q}*R_p |c_{i,q}*R_p    |R_{p*q}            |C_q*R_p
    C_p    |s*R_p  |r_{i,q}*C_p    |c_{p(i),q}*C_p |R_q*C_p            |C_{p*q}
    
