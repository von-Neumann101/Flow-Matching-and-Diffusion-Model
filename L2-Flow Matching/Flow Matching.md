本节的目标是**训练**Flow Model——确定一个好的参数$\theta$，以使得输出的分布尽可能服从$p_{data}$
![[Pasted image 20260605145547.png|594]]
我们只知道数据$x_1,x_2,\ldots,x_N\sim p_{data}$，除此以外我们对这个分布一无所知
# Probability Paths
![[Pasted image 20260605144410.png]]
$x$是一个向量（数据点）
Dirac Distribution：$x\sim\delta_z\Longleftrightarrow x=z$（没有随机，没有概率，只是把等式当成一个分布）
## Conditional Probability Paths
**条件概率路径**：一个关于$x$和$z$的函数$p_t(x\mid z)\qquad 0\le t\le 1,\ x,z\in\mathbb{R}^d$，满足如下条件：
- 给定$z$的条件概率路径是一个概率分布
- $p_0(\cdot\mid z)=\underbrace{p_{init}}_{与数据点z无关}$，且$p_1(\cdot\mid z)=\delta_z$
 ![[Pasted image 20260605151141.png|524]]
条件路径就是把一个分布坍缩为一个点（给定的$z$）
Example——Gaussian prob path
- Noise schedulers：随时间变化的系数$\alpha_t,\beta_t\in\mathbb{R}，满足\alpha_0=\beta_1=0,\ \alpha_1=\beta_0=1$
我们将**高斯概率路径**（常用）定义为$p_t(\cdot\mid z)=\mathcal{N}(\alpha_tz,\beta_t^2 I_d)$
>Proof.
>给定一个z，这是一个高斯分布（是一个概率分布）
>t=0的时候是标准高斯分布，t=1的时候是$\mathcal{N}(z,0)=\delta_z$
## Marginal Probability Paths
不是针对单个数据点——我们让数据点$z$变为随机的，不再关心$z$
**边缘化(marginalization)：全概率公式的连续版本**
**边缘概率路径**：一个关于$x$的函数$p_t(x)$是一个分布，满足
- $p_0=p_0(x)$和$z$无关，$p_1=p_{data}$

先$z\sim p_{data}$，然后通过条件概率路径$x\sim p_t(\cdot\mid z)$采样得到$x$，最后固定每一个$z$，得到$p_t(x)$，也就是：
$$
p_t(x)=\int p_t(x\mid z)p_{data}(z)dz
$$
对于每个数据点$z\sim p_{data}$，按照概率收束世界线为一条
## 对比
![[Pasted image 20260605155021.png]]
所以，probability path 描述的是“分布如何连续地从一个形态变成另一个形态”（Noise->Data）
# Vector Field
## Conditional Vector Field
**条件向量场**（对应ODE的VF）是由一个函数定义$u_t^{target}(x\mid z)\in\mathbb{R}^d$向量场，使得对于一个$x_0\sim p_{init}$，并对其进行ODE演化，能让这个$x$跟随**条件概率路径**($X_t\sim p(\cdot \mid z)$)——对应ODE的Trajectory
$$
X_0\sim p_{init},\ dX_t=u_t^{target}(X_t\mid z)dt\Longrightarrow X_t\sim p_t(\cdot\mid z)
$$
![[Pasted image 20260605160222.png|275]]![[Pasted image 20260605160344.png|309]]
最终坍缩为一个点（条件概率路径）
目前我们在做的事情——如何从一个高斯噪声分布变为一个**已知数据点**
## Marginal Vector Field
$$
u_t^{target}(x)=\int u_t^{target}(x\mid z)\underbrace{\frac{p_t(x\mid z)p_{data}(z)}{p_t(x)}}_{由贝叶斯得到的后验分布}dz
$$
这里是对路径的加权求和：现在有很多（无穷——因为是积分）的数据点$z$，每个$x$在一个条件VF被吸引到$z$的程度，我们对这个进行加权求和
![[Pasted image 20260605162431.png|177]]
那么他有如下的**重要性质**
$$
X_0\sim p_{init},\ dX_t=u_t^{target}(X_t)dt\Longrightarrow X_t\sim p_t(x)，特别地：X_1\sim p_{data}
$$一样的，这里是从初始分布沿着边缘概率路径到数据分布
Marginal Vector Field 的作用就是定义一个 ODE，把初始分布$p_{\text{init}}$推到数据分布 $p_{\text{data}}$——这正是我们要学习的
### Continuity Equation
证明：Marginal Vector Field使得$p_{\text{init}}$变为$p_{\text{data}}$
给定了一个ODE，有等价关系
$$
X_t\sim p_t\Longleftrightarrow\frac{d}{dt}p_t(x)=-\mathrm{div}(p_tu_t)(x)
$$
在MVF的演变下，每个时刻$X_t\sim p_t$，当且仅当**Continuity Equation**成立，接下来只需要证明该微分方程成立即可：
积分算子，微分算子，nabla算子此处可交换
$$
\frac{d}{dt}p_t(x)=\frac{d}{dt}\int p_t(x\mid z)p_{data}(z)dz=\int \frac{d}{dt} p_t(x\mid z)p_{data}(z)dz
$$
***注意：我们对$u_t^{target}(x\mid z)$的定义是使得分布能按照CPP到$z$的VF，所以他自然满足连续性方程。此外，课上在这里的写法是错误的，因为$p_t(\cdot \mid z)$是分布而不是函数***
$$
=-\int\mathrm{div}(p_tu_t^{target})(x \mid z)p_{data}(z)dz=-\nabla\cdot\int p_t(x\mid z)u_t^{target}(x\mid z)p_{data}(z)dz
$$
注意到MVF的定义式（$p_t(x)$和$z$无关）：
$$
=-\nabla\cdot p_t(x)\int u_t^{target}(x\mid z)\frac{p_t(x\mid z)p_{data}(z)}{p_t(x)}dz=-\nabla\cdot p_t(x)u_t(x)
$$
# Flow Matching
“学习Marginal Vector Field”
目标：训练一个Neural  Net：$u_t^\theta(x)$，使得$u_t^\theta\approx u_t^{target}$

Flow Matching的损失函数为：
$$
L_{FM}(\theta)=\mathbb{E}[||u_t^\theta(x)- u_t^{target}(x)||^2]
$$
我们注意到这里有部分是不可计算的（它要把所有$z\sim p_{data}$的条件路径混合），但是我们知道一个性质，使得损失函数最小的解恰好就是$u_t^{target}$（这里确实看着像废话，我们接着往下看）
我们**唯一能做**的是——求$u_t^{target}(x\mid z)$——之所以它能求，是因为**条件概率路径好求**（准确来说，并非好求，而这是可以人为定义的，比如高斯概率路径）
所以我们写为**Conditional FM Loss**
$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t,z,x}
\left[
\left\|
u_t^\theta(x)-u_t(x\mid z)
\right\|^2
\right]\qquad z\sim p_{data},\ x\sim p_t(\cdot\mid z),\ t为0到1之间的随机数
$$
他们满足：
$$
L_{\mathrm{CFM}}(\theta)+C=L_{\mathrm{FM}}(\theta)，其中C为和\theta无关的常数
$$
其中
$$
u_t(x)
=
\mathbb{E}
\left[
u_t(x\mid Z)\mid X_t=x
\right].
$$
于是
$$
\begin{aligned}
L_{\mathrm{CFM}}(\theta)
&=
\mathbb{E}
\left[
\left\|
u_t^\theta(x)-u_t(x\mid z)
\right\|^2
\right]
\\
&=
\mathbb{E}
\left[
\left\|
u_t^\theta(x)-u_t(x)
+
u_t(x)-u_t(x\mid z)
\right\|^2
\right]
\\
&=
\mathbb{E}
\left[
\left\|
u_t^\theta(x)-u_t(x)
\right\|^2
\right]
+
\mathbb{E}
\left[
\left\|
u_t(x)-u_t(x\mid z)
\right\|^2
\right]
\\
&\quad
+
2\mathbb{E}
\left[
\left(
u_t^\theta(x)-u_t(x)
\right)
\cdot
\left(
u_t(x)-u_t(x\mid z)
\right)
\right].
\end{aligned}
$$
交叉项为
$$
\begin{aligned}
&\mathbb{E}
\left[
\left(
u_t^\theta(x)-u_t(x)
\right)
\cdot
\left(
u_t(x)-u_t(x\mid z)
\right)
\right]
\\
&=
\mathbb{E}_{t,x}
\left[
\left(
u_t^\theta(x)-u_t(x)
\right)
\cdot
\mathbb{E}
\left[
u_t(x)-u_t(x\mid Z)
\mid t,x
\right]
\right]
\\
&=
\mathbb{E}_{t,x}
\left[
\left(
u_t^\theta(x)-u_t(x)
\right)
\cdot
0
\right]
\\
&=0.
\end{aligned}
$$
因此
$$
L_{\mathrm{CFM}}(\theta)
=
L_{\mathrm{FM}}(\theta)
+
\mathbb{E}
\left[
\left\|
u_t(x)-u_t(x\mid z)
\right\|^2
\right].
$$
令
$$
C'
=
\mathbb{E}
\left[
\left\|
u_t(x)-u_t(x\mid z)
\right\|^2
\right],
$$
则
$$
L_{\mathrm{CFM}}(\theta)
=
L_{\mathrm{FM}}(\theta)+C',
$$
其中 $C'$与 $\theta$ 无关。
