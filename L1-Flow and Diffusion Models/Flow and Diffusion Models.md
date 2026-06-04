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
- Random init cond：$X_0\sim p_{init}= \mathcal{N}(0,I)$
- ODE：$\dot X_t=u_t^{\theta}(X_t)$
目标：使得终点$X_1\sim p_{data}$
![[Pasted image 20260604171302.png|410]]![[Pasted image 20260604171346.png|218]]![[Pasted image 20260604171356.png|293]]
