# Score Matching
Flow Model的另外一种训练方法
![[Pasted image 20260606150539.png|607]]
**Conditional Score**：$\nabla_x\log p_t(x\mid z)$
**Marginal Score**：$\nabla_x\log p_t(x)$
二者之间的关系为：
$$
\nabla_x \log p_t(x)=\int\nabla_x\log p_t(x\mid z)\frac{p_t(x\mid z)p_{data}(z)}{p_t(x)}dz
$$
早期的Diffusion Model学习Score Function，至少对于高斯概率路径来说，Score Function只是Vector Field的重参数化。也就是说SF和VF是等价的

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
