The function
$$l(x)=(-2x+5)/9$$
is a solution of 
$$2f(3-2x)+f(\frac{3-x}{2})=x \tag{1}$$
as [this answer](https://math.stackexchange.com/a/5112969/11206) pointed out. And if $f(x)$ is a solution of $(1)$ then 
$$g(x):=f(x)-l(x)$$
is a solution of
$$2g(3-2x)+g(\frac{3-x}{2})=0 \tag{2}$$
and after setting 
$$h(x):=g(\frac{3-x}{2})$$
we see that 
$$2h(2x)+h(x)=0 \tag{3}$$
and this means

$$h(x)=-2h(2x) \tag{4}$$
$$h(x)=-h(x/2)/2 \tag{5}$$
So if $j$ is an arbitrary function
$$j:(-2,-1]\cup {0} \cup  [1,2)  -> \mathbf{R} $$
then $j$ satisfies ${3}$ on its domain and by $(4)$ it can be uniquely extended to the domain 

$$(-4,2] \cup (-2,-1]\cup {0} \cup  [1,2) \cup  [2,4)$$
and by $(5)$ further to
$$(-4,2] \cup (-2,-1] \cup (-1, -\frac 1 2]\cup {0} \cup [\frac 1 2, 1 ) \cup  [1,2) \cup  [2,4)$$
which is 
$$(-4,-\frac 1 2]\cup {0} \cup [\frac 1 2 ,4)$$
We can extend an arbitraty function $j$ from $(-2,-1]\cup {0} \cup  [1,2)$ to $\mathbf{R}$ inductively in a unique way and each $h$ that satisfies $3$ can be constructed in such a way.
If we substitute back we see that
$$h(-2)=g(\frac 5 2) \\
h(-1)=g(2) \\
h(0)=g(\frac 3 2) \\
h(1)=g( 1) \\
h(2)=g(\frac 1 2)
$$
so an arbitrary  function 
$$k: (\frac 1 2,1] \cup \{\frac 3 2\}\cup [ 2, \frac 5 2)\to \mathbf{R}$$
can be extended to a function $g: \mathbf{R} \to  \mathbf{R}$. And so there are functions 
$$f(x)=g(x)+\frac{-2x+5}9$$
that are not injective, e.g. if we choose a function  $g(x)$ with 
$$g(1)=-1/3$$
and $$g(2)=-1/9$$
because then we have 
$$f(1)=f(2)=0$$
and so $f$ is not injective.
