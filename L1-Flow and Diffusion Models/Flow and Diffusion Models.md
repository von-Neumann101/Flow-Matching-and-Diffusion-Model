# From Generation to Sampling
## 数学形式化
1. 数据
![[Pasted image 20260604155123.png]]
我们以向量作为模型的输入输出
2. 结果
如果我要求模型生成一个狗的照片，我们形式化为——在这个提示词的条件下，图像的分布在哪一个图像的概率上最大
![[Pasted image 20260604155402.png]]
## 分布
如果我们从一个狗的数据集里采样$z \sim p_{dog}$ ，那么z为一张狗的概率很高
> [!NOTE] 概率密度函数
> 概率密度函数输入一个数据 $z$，告诉你：如果从该分布中采样，采到 $z$ **附近**的数据的可能程度有多大
> 和概率的关系：
> $$
> P(Z\in A)=\int_Ap(z)dz
> $$
> 概率密度是高度，概率是密度在某个区域上的积分

如果我们需要指明我们需要什么数据，这就是一个条件分布，y就是输入的提示词
$$
z \sim p_{data}(\cdot | y)
$$
## 模型
![[Pasted image 20260604162444.png]]
# Flow and Diffusion Models
## Flow Model
Flow Model依赖于常微分方程
**Trajectory**：$X:\ [0,1]\to\mathbb{R}^d,\ t\mapsto X_t$，表示轨迹（时间的函数）
**Vector Field**：为某个位置赋予一个向量
ODE（常微分方程）：
给出轨迹的**初始条件**，以及轨迹应该如何**演化**，分别为：
$$
{ X _ { 0 } = x _ { 0 }, \qquad \frac { d X _ { t } } { d t } } = v _ { t } ( X _ { t } )
$$
**Flow**：每一个Flow对应一个ODE
$$
\begin{aligned}
\psi:\mathbb{R}^d \times [0,1] &\to \mathbb{R}^d, \\
(x_0,t) &\mapsto \psi_t(x_0).
\end{aligned}
$$
$$
\frac{d}{dt}\psi_t(x_0)
=
v_t\bigl(\psi_t(x_0)\bigr),
\qquad
\psi_0(x_0)=x_0.
$$
```python
def flow(x0):
    def trajectory(t):
        return position_of_x0_at_time_t
    return trajectory
```
Flow 可以看成一个输入初始位置、输出运动轨迹函数的高阶函数
固定 $x _ { 0 }$   $t \mapsto \psi _ { t } ( x _ { 0 } )$ 是一条trajectory
固定 $t$    $x _ { 0 } \mapsto \psi _ { t } ( x _ { 0 } )$ 是一个 flow map
下图就是一个Flow例子，固定一个t，x0为变量
![[Pasted image 20260604165056.png|332]]![[Pasted image 20260604165025.png|227]]

![[Pasted image 20260604170000.png]]
显然，我们几乎不能得到一个解析解，我们通常用数值解
![[Pasted image 20260604170226.png|530]]
这里我们通过选择一个步数，进而确定步长，对这个局部进行线性近似。和学习率一样：
![[Pasted image 20260604170326.png]]

Flow Model：使用ODE，使得 $p_{init}\to p_{data}$
- Neural Network: $u_t^{\theta}$是一个**Vector Field**（其中$\theta$是参数）
- Random init cond：$X_0\sim p_{init}= \mathcal{N}(0,I_d)$
- ODE：$\dot X_t=u_t^{\theta}(X_t)$
目标：使得终点$X_1\sim p_{data}$
![[Pasted image 20260604171302.png|410]]![[Pasted image 20260604171346.png|218]]![[Pasted image 20260604171356.png|293]]
## Diffusion Model
将Flow Model推演到Diffusion Model，我们使用的数学工具从ODE变为了SDE(Stochastic Differential Equations)
还是从之前描述常微分方程的两个东西来推广
- Trajectory：$X_t(随机变量)\in\mathbb{R}^d$，不再是一个确定的了
- Vector Field：和之前一样
- Diffusion Coefficient：$\sigma[0,1]\to\mathbb{R}_{\ge 0},\quad t\mapsto \sigma$（噪声的大小）
注意：扩散系数是固定的
在给出SDE之前，我们先重写一下ODE：
$$
 \frac { d X _ { t } } { d t }  = v _ { t } ( X _ { t } )\ \Longleftrightarrow \ dX_t=v_t(X_t)dt\ \Longleftrightarrow \ X_{t+h}=X_t+hu_t(X_t)+R_t(h)，其中\lim_{h\to0}\frac{R_t(h)}{h}=0
$$
现在给出SDE：
 $$
 X_0=x_0,\qquad X_t=\underbrace{du_t(X_t)dt}_{\mathrm{ODE}}+\underbrace{\sigma_tdW_t}_{\mathrm{Brownian\ motion}}
$$
**Brownian motion（布朗运动）**：是一个随机过程——$W_t\in \mathbb{R}^d$
1. $W_0=0$
2. 高斯增量：$W_t-W_s \sim \mathcal{N}(0,(t-s)I_d)\ (t>s)$方差随时间线性变化
3. 独立增量：对于任意时间划分$0\le t_1<t_2<\ldots<t_N$，$W_{t_1}-W_{t_0},\ W_{t_2}-W_{t_1},\ldots,\ W_{t_n}-W_{t_{n-1}}$是相互独立的随机变量
模拟：$W_{t+h}=W_t+\sqrt{h}\varepsilon\ (\varepsilon\sim \mathcal{N}(0,I_d))$，计算机不能存储连续的位置，所以对每个时间步$h$采样
我们重写SDE（对应ODE的重写）：
$$
X_{t+h}=X_t+hu_t(X_t)+\sigma_t(W_{t+h}-W_t)+R_t(h)，其中\lim_{h\to0}\frac{\mathbb{E}[||R_t(h)||^2]}{h}=0
$$
其中
$$
\lim_{h\to0}\frac{\mathbb{E}[||R_t(h)||^2]}{h}=0\Longleftrightarrow\mathbb{E}[||R_t(h)||^2]^\frac{1}{2}=o(\sqrt h)
$$**误差项可以忽略**
![[Pasted image 20260604204836.png|625]]
很直观——ODE+随机噪声（$\sigma=0$就是ODE了）

可以说随机噪声引入了一种探索的机制，ODE的轨迹确定了就确定了，但是SDE可以进行一些探索从而产生一些优化

Diffusion Model：和Flow Model的目的一样，只不过多了随机过程
# 总结
![[Pasted image 20260605140303.png]]
最终我们输出的是一个分布$X_1$