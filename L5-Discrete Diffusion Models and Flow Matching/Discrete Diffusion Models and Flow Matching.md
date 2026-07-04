![[Pasted image 20260610095338.png]]
离散的流匹配与扩散模型用于**序列数据**（离散空间里没有SDE和ODE）——CTMCs(Continuous-time Markov chains)
# CTMCs
**词表(Alpha bet)**：$V=\{v_1,...,v_N\}$
**状态空间**：$S=V^d$，其中的点$x\in S$是一个字符序列
**Rate Matrix**：$Q:S\times S\times [0,1]\to \mathbb R_{\ge 0}$，$Q_t(x\mid y)$理解为状态$x$**切换**为状态$y$的速率（离散）
- $Q_t(x\mid x)=-\sum_{y\ne x}Q_t(y\mid x)$，也就是说对$Q$的列求和（x是列索引）的结果为0
**CTMCs**：无记忆的随机过程$X_t\in S$，有ODE：
$$\frac{d}{dh}p_{t+h|t}(X_{t+h}=y\mid X_t=x)|_{h=0}=Q_t(y\mid x)$$
![[Pasted image 20260610114519.png]]
**CTMC model**：$Q_t^{\theta}(y\mid x)$
**Factorization Condition**：$Q_t^{\theta}(y\mid x)=0$，当x和y在超过一个索引处不同。即CTMC每次只能改一个位置
```
x: 010100
y: 010000
z: 011000
```
**Neural Network**：
$$
(x,t)\stackrel{forward}{\longrightarrow}(Q_t^\theta(y_{ij}\mid x))_{y_i\in V,\ j=1,2, ...,d}=\left( \begin{matrix} { Q _ { t } ^ { \theta } ( v _ { 1 }, 1 | x ) } & { \cdots Q _ { t } ^ { \theta } ( v _ { V }, 1 | x ) } \\ { \cdots } \\ { Q _ { t } ^ { \theta } ( v _ { 1 }, d | x ) } & { \cdots Q _ { t } ^ { \theta } ( v _ { V }, d | x ) } \\ \end{matrix} \right)
$$
（$d$是每个数据的维度，$V$是数据每个维度可以取的值的集合）
例如，$Q_t^\theta(0,4\mid x)$表示：把x的第4位改为0的转移速率。完整的状态转移矩阵的每一个元素给出**只考虑x改一个位形成的元素，如果修改完以后和原来只有一个位有区别那么就给出速率**。由Factorization Condition，我们可以感觉到**完整的状态转移矩阵是稀疏矩阵**（因为非0的全在上面的矩阵了（大小只有dV））
![[Pasted image 20260611090021.png]]**Sampling with CTMC models**：初始的分布为均匀分布
$$
\begin{align*}
&X_0\sim p_{init}\\
&X_{t+h}\sim p_{t+h|t}(y\mid X_t)
\end{align*}
$$
变换
$$
\begin{align*}
p_{t+h|t}(y\mid X_t)&\approx p_{t|t}(y\mid X_t)+h\frac{d}{dh}p_{t+h| t}(y\mid X_t)\mid_{h=0}\\
&=\delta_y(x)+hQ_t^\theta(y\mid x)
\end{align*}
$$
如果我们对y求和，结果为1，说明泰勒展开近似的式子是一个分布
![[Pasted image 20260611092005.png|485]]
# Discrete Flow Matching Matrix
**Conditional Prob path**：$p_t(x\mid z)\quad x,z\in S$是关于$x$的概率密度函数
$$
p _ { 0 } ( \cdot | z ) = p _ { \mathrm { i n i t } }, \quad p _ { 1 } ( \cdot | z ) = \delta _ { z }
$$
**Marginal Prob path**：$p _ { t } ( x ) = \sum _ { z \in S } p _ { t } ( x | z ) p _ { \mathrm { d a t a } } ( z )$，这里就是离散的了：
$$
\begin{align*}
p_0= p_{init}\\
p_1=p_{data}
\end{align*}
$$
**Example**：
![[Pasted image 20260619201534.png]]
此处的$x$就是离散的输入序列，同时我们假设每个token之间都为独立的（第一位为token1、第二位为token2、......、第N位为tokenN），然后我们对与每个独立的token做插值。
对于采样来说，我们以一定的概率让其为**数据或者噪声之一**

**Conditional Rate Matrix**：$$