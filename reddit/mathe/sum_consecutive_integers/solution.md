<pre>
N = m + (m+1) + ... + (n-1) + n 
  = (1 + 2 + ... + (n-1) + n) - (1 + 2 + ... + (m-2) + (m-1)) 
  = (n * (n+1))/2 - ((m-1) * m)/2 
  = (n + m) * ( n -m + 1) /2         (1)
</pre>
So we have to solve
<pre>
  (n + m) * ( n -m + 1) = 2 * N      (2)
</pre>
which can be split into 
<pre>
  a = n + m                          (3)
  b = n - m +1
  </pre>
  where 
  <pre>
    a*b = 2*N                        (4)
  </pre>
  N and m are positive integers and n >= m, so a and b are positive integers, too,and of opposite parity. We have a>b. We get
  <pre>
    n = (a + b -1) / 2               (5)
    m = (a - b +1) / 2
  </pre>
a+b-1 and a-b+1 are either both odd or both even and so that they are divisible by 2. This happens if one of a or b i odd and the other oen is even. 

If we select an odd divisor d of N, then 2\*N/d is even. We set 
<pre>
  a = max{d, 2*N/d}                  (6)
  b = min{d, 2*N/d}
</pre>
 and can calculate n and m by equations (5). On the other hand, if we have given n and m, n>=m, then we can calculate a and b by (3). One of these two numbers is odd and this is the corresponing odd divisor of N.

  Note that for the odd divisor 1 of N we get b=1 and a=2\*N from (5) we get n=N and m=N and therefore N can be expressed by the sum
  <pre>
    N                               (7)
  </pre>
  This is a sum with only one summand. A sum with only one summand is not unusual in mathematics. So from the above follow the following statements:
1. Every natural number N can be represented in t(N) ways as the sum of a sequence of consecutive positive integers, where t(N) is the number of odd divisors of N. For each odd divisor d of N, the corresponding values ​​for m and n can be calculated using (6) and (5).
2. Powers of two have only one odd divisor, namely 1. The corresponding sum is the sum with the single summand N.
3. The sequence a(N), where a(N) is the number of different sequences of consecutive positive integers that sum up to N, is not bounded. 

Statement 1 follows from the preceding explanations. Statement 2 follows from statement 1. And statement 4 also follows from statement 1 and the fact that the sequence t(N) of odd divisors is not bounded



