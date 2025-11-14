https://math.stackexchange.com/questions/5108663/sum-of-sum-k-1-infty-frac-1k-1k2-correct-to-five-decimal-pla

Here we have an alternating series 

$$s_n=\sum_{k=1}^n (-1)^na_k$$

where the values of the sequence $a_n$ are strictly decreasing. So the partial sums have the following property

$$s_1 \lt s_3 \lt s_5 \lt \ldots \lt s_\infty \lt \ldots \lt s_6 \lt s_4 \lt s_2 \tag{1}$$

and so the error  

$$\mathrm{err}:=|s_\infty-s_n|<|s_{n+1}-s_n|=a_{n+1}=(n+1)^{-2}$$.

But this is only an upper bound for the error.


If we define 

\begin{align}
L_n&:=&s&_{2n-1}\\
U_n&:=&s&_{2n}
\end{align}

we can rewrite $(1)$ as

$$L_1\lt L_2 \lt \ldots \lt s_\infty \lt ... \lt U_2 \lt U1 $$
If we set
\begin{align}
l_n:=L_n-L{n-1}
u_n:=U_n-U_{n-1}
\end{align}

we see that 
$$l_1 \lt u_1 \lt l_2 \lt u_2 \lt \ldots \tag{2}$$

Now let's leave the filed of 






    /* [wxMaxima: input   start ] */
    s[k+2]:(-1)^(k+1)*a_(k+2)+(-1)^(k)*a_(k+1)+(-1)^(k-1)*a_(k)+s[k-1];
    s[k+1]:                   (-1)^(k)*a_(k+1)+(-1)^(k-1)*a_(k)+s[k-1];
    s[k]:                                      (-1)^(k-1)*a_(k)+s[k-1];
    /* [wxMaxima: input   end   ] */


    /* [wxMaxima: input   start ] */
    s[k+2]-s[k];
    /* [wxMaxima: input   end   ] */


    /* [wxMaxima: input   start ] */
    1/k^2-1/(k+1)^2,ratsimp;diff(%,k),ratsimp;%,factor;
    /* [wxMaxima: input   end   ] */
