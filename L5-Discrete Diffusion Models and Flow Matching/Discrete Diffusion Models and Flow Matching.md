扩散语言模型：以任意顺序生成文本的语言模型：
![[Pasted image 20260610095338.png]]
离散的流匹配与扩散模型用于**序列数据**（离散空间里没有SDE和ODE），我们使用其他的数学工具——CTMCs(Continuous-time Markov chains)
# CTMCs
**词表(Alpha bet)**：$V=\{v_1,...,v_N\}$
**状态空间**：$S=V^d$，其中的点$x\in S$是一个字符序列
**Rate Matrix**：$Q:S\times S\times [0,1]\to \mathbb R_{\ge 0}$（接受两个状态以及时间，生成一个非负数），$Q_t(x\mid y)$理解为状态$x$**切换**为状态$y$的速率（之所以是速率是因为状态空间离散，我们当然不关心路径）
- $Q_t(x\mid x)=-\sum_{y\ne x}Q_t(y\mid x)$，也就是说对$Q$的列求和（x是列索引）的结果为0

**CTMCs**：无记忆的随机过程$X_t\in S$，有ODE：
$$\frac{d}{dh}p_{t+h|t}(X_{t+h}=y\mid X_t=x)|_{h=0}=Q_t(y\mid x)$$
在无穷小时间内，状态x变为状态y的概率，由速率矩阵给出
**Example**：
![[Pasted image 20260610114519.png]]
**CTMC model**：$Q_t^{\theta}(y\mid x)$
**Factorization Condition**：$Q_t^{\theta}(y\mid x)=0$，当x和y在超过一个索引处不同。即CTMC每次只能改变一个位置
```
x: 010100
y: 010000
z: 011000
```
这里$Q_t^\theta(z\mid x)=0$

**Neural Network**：
$$
(x,t)\stackrel{\text{forward pass}}{\longrightarrow}(Q_t^\theta(y_{ij}\mid x))_{v_i\in V,\ j=1,2, ...,d}=\left( \begin{matrix} { Q _ { t } ^ { \theta } ( v _ { 1 }, 1 | x ) } & { \cdots Q _ { t } ^ { \theta } ( v _ { V }, 1 | x ) } \\ { \cdots } \\ { Q _ { t } ^ { \theta } ( v _ { 1 }, d | x ) } & { \cdots Q _ { t } ^ { \theta } ( v _ { V }, d | x ) } \\ \end{matrix} \right)
$$
（$d$是每个数据的维度，$V$是数据每个维度可以取的值的集合）
例如，$Q_t^\theta(0,4\mid x)$表示：把x的第4位改为0的转移速率。完整的状态转移矩阵的每一个元素给出**只考虑x改一个位形成的元素，如果修改完以后和原来只有一个位有区别那么就给出速率**。由Factorization Condition，我们可以感觉到**完整的状态转移矩阵是稀疏矩阵**（因为非0的全在上面的矩阵了（大小只有$d\times V$））
![[Pasted image 20260611090021.png]]**Sampling with CTMC models**：初始的分布为均匀分布
$$
\begin{align*}
&X_0\sim p_{init}\\
&X_{t+h}\sim p_{t+h|t}(\cdot\mid X_t)
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
## Probability Path
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
![[Pasted image 20260619201534.png|568]]
这是一种构造条件概率**路径**的方式：
此处的$x$就是离散的输入序列，同时我们假设每个token的扰动都为独立的——由随机token序列开始，部分的位置变成真实的token（对于采样来说，我们以一定的概率让其为**数据或者噪声之一**），最后变为真实的token序列（换句话说，时间为$t$时，x的分布由其每个分量插值的乘积给出）
## Rate Matrix
**Conditional Rate Matrix**：$Q_t^z(y\mid x)\quad x,y,z\in S,t\in[0,1]$，类比于Flow Matching中的ODE和Diffusion Model中的SDE，其具有如下性质：
$$
X_0\sim p_{init},\frac{d}{dh}p_{t+h|t}(X_{t+h}=y\mid X_t=x)|_{h=0}=Q_t(y\mid x)\Rightarrow X_t\sim p_t(\cdot\mid z)
$$
**Marginal Rate Matrix**：$Q_t(y\mid x)=\sum_{z\in S}Q_t^z(y\mid x)\frac{p_t(x\mid z)p_{data}(z)}{p_t(x)}$，满足：
$$
X_0\sim p_{init},用\mathrm{CTMCs}演化X_t\Rightarrow X_t\sim p_t
$$
> [!NOTE] 符号说明
> $Q_t^z(y\mid x)$中$z$代表整条条件路径的终点，而$y$只是某一次跳转的终点

我们证明边缘速率矩阵的性质：
**Kolmogorov Forward Equation**：给定一个CTMCs，有等价关系：
$$X_t\sim p_t\quad(0\le t\le 1)\Longleftrightarrow \frac d {dt}p_t(x)=\sum_{y\in S}Q_t(x|y)p_t(y)$$
证明和之前的各种边缘向量场的一样

![[Pasted image 20260707092637.png]]
**Marginal Rate Matrix for Factorized Mixture Path**：
$$
\begin{align*}
Q_t(y\mid x)&=(Q_t(v_i,j\mid x))_{v_i\in V,j=1,2,...,d}\\
Q_t(v_i,j\mid x)&=\dfrac{\dot \kappa_t}{1-\kappa_t}\left[p_{1\mid t}(z_j=v_i\mid x)-\delta_{x_j}(v_i)\right]
\end{align*}
$$
$p_{1\mid t}(z_j=v_i\mid x)$表示：给定中间状态$x$，第j位最终为$v_i$的概率
自然，我们训练的目标就是$p_{1|t}^\theta(z_j\mid x)$——由于我们要判别并给出每一个可能的$v_i$的类别概率，这是一个分类器，我们使用交叉熵损失函数作为损失函数
$$\mathcal L_{\text{DFM}}(\theta)=\mathbb E_{z\sim p_{data},t\sim\text{Unif}_{[0,1]},x\sim p_t(\cdot|z)}\left[\sum_{j=1}^d-\log p_{1|t}^\theta(z_j|x)\right]$$
![[Pasted image 20260707101618.png]]
这里5~9行是4行的具体步骤
# 优点与缺点
![[Pasted image 20260707103219.png]]