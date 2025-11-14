https://math.stackexchange.com/questions/5108663/sum-of-sum-k-1-infty-frac-1k-1k2-correct-to-five-decimal-pla

Here we have an alternating series $s_n=\sum_{k=1}^n (-1)^na_k$ where the values of the sequence $a_n$ are strictly decreasing.So the partial sums have the following property

$$s_1 \lt s_3 c s_5 \lt \ldots \lt s_\infty \lt \ldots \lt s_6 \lt s_4 \lt s_2$$

and we have 
$$|s_\infty-s_n|<|s_{n+1}-s_n|=a_{n+1}$$


    /* [wxMaxima: input   start ] */
    kill(all);
    /* [wxMaxima: input   end   ] */


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
