<pre>
N = m + (m+1) + ... + (n-1) + n 
  = (1 + 2 + ... + (n-1) + n) - (1 + 2 + ... + (m-2) + (m-1)) 
  = (n * (n+1))/2 - ((m-1) * m)/2 
  = (n + m) * ( n -m + 1) /2
</pre>
So we have to solve
<pre>
  (n + m) * ( n -m + 1) = 2 * N
</pre>
which can be split into 
<pre>
  n + m    = a
  n - m +1 = b
  </pre>
  where 
  <pre>
    a*b = 2*N
  </pre>
  N and m are positive integers and n >= m, so a and b are positive integers too. We get
  <pre>
    n = (a + b -1) / 2
    m = (a - b +1) / 2
  </pre>
  a+b and a-b are either both odd or both even. The same is therefore true for (a + b -1) and (a - b +1). The latter two terms must be even so that they are divisible by 2. This happens if one of a or b i odd and the other oen is even. m is positive, so a is not smaller than b.
