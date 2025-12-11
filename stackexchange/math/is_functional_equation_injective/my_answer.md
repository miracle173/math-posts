The function

$$l(x)=(-2x+5)/9$$

is a solution of the type

$$f: x \to ax+b$$

of 

$$  2f(3-2x)+f(\frac{3-x}{2})=x \tag{1} $$

as [this deleted answer](https://math.stackexchange.com/a/5112969/11206) pointed out. And if $f(x)$ is a solution of $(1)$ then 

$$g(x):=f(x)-l(x)$$

is a solution of

$$2g(3-2x)+g(\frac{3-x}{2})=0 \tag{2}$$

after substituting

$$y=3-2x$$

we have

$$2g(y)+g((y+3)/4)=0$$

and further

$$g(y)=-g((y+3)/4)/2 \tag{3}$$

and after substituting

$$y=(3-x)/2$$

we get

$$2g(4y-3)+g(y)=0$$

and so

$$g(y)=-2g(4y-3) \tag 4$$

If 

$$4y-3=(y+3)/4$$

then $y=1$ and the value of each side of the above quation is $1$ and therefore 

$$g(1)=-2g(1)=-g(1)/2$$

which implies 

$$g(1)=0 \tag{5}$$

The sequence

$$y_{n+1}=4y_n-3$$

is strictly monotonically decreasing to $-\infty$ if we start with a value $y_1<1$ and strictly monotonically increasing to $+\infty$, if we start with $y_1>1$. The sequence

$$y_{n+1}=(y_n+3)/4$$

is strictly increasing to $1$ if we start with $y_1<1$ and strictly decreasing to $1$ if we start with $y_1>1$.
We can partition $\mathbf{R}$ n the following way:

$$\bigcup_{n=1}^\infty [a_{n+1},a_n) \cup \bigcup_{n=1}^\infty [b_n,b_{n+1}) \cup 
\{1\} \cup 
\bigcup_{n=1}^\infty (c_{n+1},c_n] \cup \bigcup_{n=1}^\infty (d_n,d_{n+1}] $$

where

$$
\begin{array}{ccc}
a_1&=&0\\
a_{n+1}&=&-3+4a_n\\
b_1&=&0\\
b_{n+1}&=&\frac {b_n+3}4\\
c_1&=&2\\
c_{n+1}&=&\frac {c_n+3}4\\
d_1&=&2\\
d_{n+1}&=&-3+4d_n
\end{array}
$$

This partition has the property that for each $x \in \mathbf R$ the numbers $3-2x$ and $\frac{3-x}{2}$ lie in two adjacent intervals of this partition, but never in the same, except for $x=1$ where these values are both equal to $1$. Instead of the value $a_1=b_1=0$ a different start value can be chosen, it is only necessaty that the value is less than $1$. A similar statement holds for $c_n=d_n=2$.

So if $j$ is an arbitrary function

$$j:[0,b_2)\cup \{1\} \cup  (c_2,2]  \mapsto \mathbf{R} $$

then $j$ satisfies $(2)$ on its domain and by $(3)$ it can be uniquely extended to the domain 

$$[a_2,0) \cup [0, b_2) \cup \{1\}  \cup  (c_2,2] \cup  (2,d_2]$$

which is the same as 

$$[a_2,b_2) \cup \{0\} \cup (c_2, d_2)$$

Now it can be inductively extended from

$$[a_n,b_n) \cup \{0\} \cup (c_n,d_n] $$

to 

$$[a_{n+1},b_{n+1}) \cup \{0\} \cup (c_{n+1},d_{n+1}] $$

again by using $(3)$ and $(4).$

We now have extended $j$ to $\mathbf R$ consistent to $(2).$ This is the requested function $g.$ Not that every soluion $g$ of $(3)$ can be derived by this method. So  the following statements hold:

> Every function defined on $(b_2,0) \cup (c_2,0)$ that is extended to $\mathbf R$ is a solution of $(3)$. If for two solutions  $g_1$ and $g_2$of $(3)$ $g1=g_2$ holds on $[b_2,0) \cup (c_2,2]$ then $g_1=g_2$ holds on $\mathbf R.$ For every solution $g$ of $(3)$ $g(1)=0$ holds.

Now it is simple that injectivity does not follow from $(1)$. For a constant $G$ we define the function

$$
\begin{array}{rrcl}
g: \\&[0,b_2) \cup  \{1\} \cup (c_2,2] &\mapsto &\mathbf R \\
&x&\to& 
{\left\{
\begin{array}{ll}
G-l(0), &\mathrm{if}\, x=0 \\
0, &\mathrm{if}\, x=1 \\
G-l(2), &\mathrm{if}\, x=2 \\
\mathrm{arbitrary}, &x\notin \{0,1,2\}
\end{array}\right.
}\end{array}
$$





