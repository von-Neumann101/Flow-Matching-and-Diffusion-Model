# Score Matching
Flow Model的另外一种训练方法
![[Pasted image 20260606150539.png|607]]
**Conditional Score**：$\nabla_x\log p_t(x\mid z)$
**Marginal Score**：$\nabla_x\log p_t(x)$
二者之间的关系为：
$$
\nabla_x \log p_t(x)=\int\nabla_x\log p_t(x\mid z)\frac{p_t(x\mid z)p_{data}(z)}{p_t(x)}dz
$$
早期的Diffusion Model学习Score Function
这是由于Score Function只是Vector Field的重参数化。也就是说SF和VF是等价的
$$
\begin{align*}
&u _ { t } ^ { \mathrm { t a r g e t } } ( x | z ) = a _ { t } \nabla \operatorname { l o g } p _ { t } ( x | z ) + b _ { t } x\\
&u _ { t } ^ { \mathrm { t a r g e t } } ( x ) = a _ { t } \nabla \operatorname { l o g } p _ { t } ( x ) + b _ { t } x\\
&其中，a _ { t } = \left( \beta _ { t } ^ { 2 } \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } - \dot { \beta } _ { t } \beta _ { t } \right), \quad b _ { t } = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } }
\end{align*}
$$
**Score network**：$s_t^\theta(x)\in\mathbb{R}^d$
**SM loss**：$L_{SM}(\theta)=\mathbb{E}_{t,z,x}[||s_t^\theta(x)-\nabla\log p_t(x)||^2]$
**Denoising SM loss**：$L_{DSM}(\theta)=\mathbb{E}_{t,z,x}[||s_t^\theta(x)-\nabla\log p_t(x\mid z)||^2]$
同样的，他们之间满足
$$
L_{SM}(\theta)=L_{DSM}(\theta)+C\qquad C是与\theta无关的常数
$$
**SDE sampling**：
我们已知$X_0\sim p_{init},\ dX_t=u_t^{target}(X_t)dt\Longrightarrow X_t\sim p_t$
定理：（$\sigma_t\ge 0$是任意的）
$$X_0\sim p_{init},\ dX_t=[u_t^{target}(X_t)+\underbrace{\frac{\sigma_t^2}{2}\nabla\log p_t(X_t)}_{梯度场修正噪声}]dt+\sigma_tdW_t\Longrightarrow X_t\sim p_t$$
注意，这里对于任意一个扩散系数，其动力学都是一样的
**Fokker-Planck equation**
给定$X_0\sim p_{init},\ dX_t=\underbrace{[u_t^{target}(X_t)+\frac{\sigma_t^2}{2}\nabla\log p_t(X_t)]}_{u_t}dt+\sigma_tdW_t$，有以下恒等关系
$$
X_t\sim p_t\Longleftrightarrow \frac{d}{dt}p_t(x)=-\mathrm{div}(p_tu_t)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x)
$$
Laplacian算子：
$$
\Delta W_t(x)=\mathrm{div}(\nabla W_t)(x)
$$
Proof.
$$
\begin{align*}
\frac{d}{dt}p_t(x)&=\underbrace{-\mathrm{div}(p_tu_t^{target})(x)}_{连续性方程}\underbrace{-\mathrm{div}(\frac{\sigma_t^2}{2}\nabla p_t)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x)}_{=0}\\
&=-\mathrm{div}(p_tu_t^{target}+\frac{\sigma_t^2}{2}\nabla p_t)+\frac{\sigma_t^2}{2}\Delta p_t(x)\\
&=-\mathrm{div}(p_t[u_t^{target}+\frac{\sigma_t^2}{2}\nabla \log p_t])+\frac{\sigma_t^2}{2}\Delta p_t(x)
\end{align*}
$$
实际上ODE几乎就是最优，而且理论上，sigma不会改变模型的效果。但是实际上存在一个极大值点（到达该点需要用SDE）

题外话（模拟分子）
![[Pasted image 20260607101403.png|333]]
# Guidance
**Guidance**：**根据提示词生成图片**
Unguided指的是不具体指定生成图片的内容，反之则指定
## Vanilla Guidance
数据的分布$(z,y)\sim p_{data}$，这里$y$就是Prompt，从数据中读取的分布还包含了**配对的**提示词和数据
**Guided VF**：$u_t^{\theta}(x\mid y)\in\mathbb{R}^d$
**Guided FM loss**：
$$
L_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t,z,x}
\left[
\left\|
u_t^\theta(x\mid y)-u_t^{target}(x\mid z)
\right\|^2
\right]\qquad (z,y)\sim p_{data},\ x\sim p_t(\cdot\mid z),\ t为0到1之间的随机数
$$
我们只是加入了一个新的输入——Prompt，而且从公式看出来新加入的输入只不过像一个**符号**，并没有从公式关系中看出一个Prompt和Data之间的某种联系
## Classifier Guidance
贝叶斯公式给出（此处的$p_t$发生了方法重载）
$$
p_t(x\mid y)=\frac{p_t(y\mid x)p_t(x)}{p_t(y)}
$$
我们对等式左右两边取Score Function
$$
\nabla_x\log p_t(x\mid y)=\nabla_x\log p_t(y\mid x)+\nabla_x\log p_t(x)
$$
我们知道评分函数和向量场等价，得到：
$$
\underbrace{u_t^{target}(x\mid y)}_{Guidance\ vector\ field}=u_t^{target}(x)+\underbrace{a_t\nabla\log p_t(y\mid x)}_{Classifier}
$$
$p_t(y\mid x)$的含义是——给定图片$x$，符合他的$y$的概率有多大
![[Pasted image 20260608084038.png|257]]
## Classifier-Free Guidance(CFG)
![[Pasted image 20260608084149.png|481]]
**Prompt-Reinforced VF**：我们让提示词对应的向量增大（无条件），得到提示词强化向量场
$$
\tilde{u}_t^w(x\mid y)=u_t^{target}(x)+wa_t\nabla_x\log p_t(y\mid x)=u_t^{target}(x)+w(u_t^{target}(x\mid y)-u_t^{target}(x))
$$
这里的变形很重要， 这决定了是否为**Classifier-Free**，如果使用前者，我们需要训练一个Classifier（不能用现成的，因为没有现成的分类器能把噪声图片推到符合提示词的地方），这增加了工作量。不过还不够——显然$u_t(x\mid y)$和$u_t(x)$不是一个东西，用不了一个神经网络，我们再做一个非常简单的变换
$$
\tilde{u}_t^w(x\mid y)=wu_t^{\theta}(x\mid y)+(1-w)u_t^{\theta}(x\mid \emptyset)
$$
**把边缘向量场看作 *没有输入提示词（提示词为空）* 的Guidance VF，这样就完全不需要Classifier了，只需要一个神经网络即可**
对比：
![[Pasted image 20260608090731.png]]
算法：
![[Pasted image 20260608090831.png|582]]
![[Pasted image 20260608091007.png|585]]
**注意到这里CFG需要调用两次神经网络——一次无提示词，一次有提示词**

**CFG**（尤其是在$w>1$的时候）不再拟合一个所谓的$p_{data}(x\mid y)$了，它在拟合一个人类的**经验数据**（来源于我们放大了Prompt的作用）——不是从严格的概率建模原则自然推出的最优方法，而是一个**实际效果很好**的技巧
![[Pasted image 20260608092100.png]]
这里让中间的模糊部分坍缩了，把分布压到更强烈符合Prompt的位置


弹吉他的熊猫：
![[Pasted image 20260608092557.png|597]]