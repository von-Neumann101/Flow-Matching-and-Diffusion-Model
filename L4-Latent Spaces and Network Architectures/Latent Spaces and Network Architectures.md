# Latent Spaces
将数据转化到潜空间，使得其被高效建模
## 问题
对于一个1080x720的彩色图像，需要一个1080x720x3的一个非常高维的向量来表示
- GPU显存不够
- 难以学习
- 大量像素的数据冗余
但是对于CNN来说，我们同样会输入一个高分辨率的彩色图片，为什么到这里就不行呢？——Diffusion Model在采样的时候要**迭代多次**，这使得和CNN输入相同的图片时，代价是几倍
## 实现
![[Pasted image 20260609081806.png]]
**Latent Space**：$z\in\mathbb{R}^k$（这是一个低维空间），接下来我们用$x$表示高维数据空间的数据点
**Encoder**：$\mu_\phi:\mathbb{R}^d\to\mathbb{R}^k$表示一个神经网络
**Decoder**：$\mu_\theta:\mathbb{R}^k\to\mathbb{R}^d$
**Reconstruction Loss**：字面意思，看经过这么一趟转换是否保持不变
$$
L_{R}(\theta,\phi)=\mathbb{E}_{x\sim p_{data}}[||\mu_\phi(\mu_\theta(x))-x||^2]
$$
**Latent Distribution**：$x\sim p_{data},\ z=\mu_\phi(x)\Longrightarrow z\sim p_{latent}$
我们的任务就是做一个好的AutoEncoder，使得获得一个好的潜空间（连续）
### Variational AutoEncoder(VAEs)
转换为Stochastic Encoding
**Encoder**：$q_\phi(z\mid x)=\mathcal{N}(z;\mu_\phi(x),\sigma_\phi^2(x)I_k)$
**Decoder**：$p_\theta(x\mid z)=\mathcal{N}(x;\mu_\theta(x),\sigma^2I_d)$
**Encode**：$z\sim q_\phi(\cdot\mid x)$
**Decode**：$x\sim p_\theta(\cdot\mid z)$
**Loss**（要让似然尽可能大，这样就能让编码出的隐变量尽可能转换为原来的数据）：
$$
L_{R}(\theta,\phi)=\mathbb{E}_{x\sim p_{data},\ z\sim q_\phi(x)}[-\log p_\theta(x\mid z)]=\mathbb{E}_{x\sim p_{data},\ z\sim q_\phi(x)}[\frac{1}{2\sigma^2}||x-\mu_\theta(z)||^2]
$$
我们要利用这个方法，使得Latent Space近似高斯分布——这是因为Flow Model 的**起点是高斯噪声**所以Autoencoder 的 latent target 也最好被规整到一个接近高斯、连续、无太多空洞的空间里。这样 Flow 从高斯噪声生成 latent 时，任务会更稳定。
**KL散度**（衡量两个分布的距离）：
$$
D _ { \mathrm { K L } } ( q ( x ) \parallel p ( x ) ) = \int q ( x ) \operatorname { l o g } { \frac { q ( x ) } { p ( x ) } } = \mathbb { E } _ { X \sim q } \left[ \operatorname { l o g } { \frac { q ( X ) } { p ( X ) } } \right].
$$
对于高斯分布：
$$
D _ { \mathrm { K L } } ( q \parallel p ) = \frac { 1 } { 2 } \left( \mathcal { K } \left( \frac { \sigma _ { q } ^ { 2 } } { \sigma _ { p } ^ { 2 } } \right) + \frac { \| \mu _ { q } - \mu _ { p } \| ^ { 2 } } { \sigma _ { p } ^ { 2 } } \right), \quad \mathrm { w h e r e ~ } \mathcal { K } ( \alpha ) = \sum _ { i } \alpha _ { i } - \operatorname { l o g } \alpha _ { i } - 1.
$$
**Prior Loss**：
$$
L_{prior}(\phi)=\mathbb{E}_{x\sim p_{data}}[D_{\mathrm{KL}}(q_\phi(\cdot\mid x)\parallel\mathcal{N}(0,I_))]=\mathbb{E}_{x\sim p_{data}}[\frac { 1 } { 2 } \left( \mathcal { K } \left(  { \sigma _ { \phi }(x) ^ { 2 } }  \right) +  { \| \mu _ { \phi }(x)\| ^ { 2 } }  \right)]
$$
注意到这个Loss的最小化要求整个q变为标准正态分布，使得q失去重构的能力。所以我们综合两个Loss
**VAE loss**：
$$
L_{\mathrm{VAE}}=L_{R}(\theta,\phi)+\beta L_{prior}
$$
注意直接把采样出来的 $z\sim q_{\phi}(\cdot \mid x)$ 送进 Decoder，**反向传播没法直接对“采样操作”求梯度**（说白了，如果你在Pytorch里把采样操作写到计算图里，程序会报错），所以我们要删去“采样操作”。
$$
q_\phi(z\mid x)=\mathcal{N}(z,\mu_\phi(x),\sigma_\phi^2(x)I_k)\Longrightarrow z=\mu_\phi(x)+\sigma_\phi^2(x)\varepsilon，其中\varepsilon\sim\mathcal N(0,I_k)
$$
得到最终可训练的**VAE Loss**
$$
L_{\mathrm{VAE}}(\theta,\phi)  
=  
\mathbb{E}_{x\sim p_{\mathrm{data}},\ \varepsilon\sim\mathcal{N}(0,I)}  
\left[  
\frac{1}{2\sigma^2}  
\left\|  
x-\mu_\theta\!\left(  
\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon  
\right)  
\right\|^2  
+  
\frac{1}{2}  
\left(  
\mathcal{K}\!\left(\sigma_\phi(x)^2\right)  
+  
\left\|\mu_\phi(x)\right\|^2  
\right)  
\right].
$$
这下就可以反向传播了
![[Pasted image 20260609093143.png|680]]
![[Pasted image 20260609095706.png]]
**总结一下，Latent Space只是一个小trick，他并不是用于解决理论上的什么东西，而是用于解决因为图片过大模型训练、输出的成本高的问题。整个Latent Diffusion/Flow Model都在Latent Space上运行——他们输入的图片由Encoder变为Latent，他们输出的图片由Decoder变为真实图片。**
# Neural Network Architectures
$$u_t^\theta(x\mid y)$$
我们如何在实际上做到这一点？
- Time
$$
{ \mathrm { T i m e E m b } } ( t ) = { \sqrt\frac { 2 } {  { d } } } \left[ \cos ( 2 \pi w _ { 1 } t ) \quad \cdots \quad \cos ( 2 \pi w _ { d / 2 } t ) \quad \sin ( 2 \pi w _ { 1 } t ) \quad \cdots \quad \sin ( 2 \pi w _ { d / 2 } t ) \right] ^ { T }
$$
频率由以下公式给出
$$
w _ { i } \; = \; w _ { \mathrm { m i n } } \left( { \frac { w _ { \mathrm { m a x } } } { w _ { \mathrm { m i n } } } } \right) ^ { \frac { i - 1 } { d / 2 - 1 } }, \qquad i = 1, \ldots, d / 2.
$$
> [!NOTE] $\mathrm { T i m e E m b }$
> 和其他的Embedding模型一样，不要考虑每个分量的具体含义，具体的含义是由模型理解的。这里唯一的作用是把一维时间 $t$ 变成一个高维、稳定、可区分的时间条件向量——Flow/Diffusion Model只在意**当前处在生成过程的哪个阶段，以及和其他时间阶段有多接近**
- Prompt
![[Pasted image 20260609105958.png|630]]
- Image
先把图片展平为Patch的序列，对于Latent Space也是一样的方法
![[Pasted image 20260609110110.png|663]]
再送入Diffusion Transformer(DiT)
1. 输入：
$$
\begin{array} { l l } { { \tilde { t } } = \mathrm { T i m e E m b } ( t ) \in \mathbb { R } ^ { k } } & { { \tilde { x } } _ { 0 } = \mathrm { P a t c h E m b } ( x ) \in \mathbb { R } ^ { N \times k } } \\ { { \tilde { y } } = \mathrm { P r o m p t E m b } ( y ) \in \mathbb { R } ^ { S \times k } } & { } \\ \end{array}
$$
2. Attention 循环：
$$
\tilde { x } _ { i + 1 } = \mathrm { D i T B l o c k } ( \tilde { x } _ { i }, \tilde { t }, \tilde { y } ) \in \mathbb { R } ^ { N \times k } \quad ( i = 0, \ldots, N - 1 )
$$
3. Unpatchify：
$$
u = \mathrm { U n p a t c h i f y } ( \tilde { z } _ { N } \tilde { W } ) \in \mathbb { R } ^ { C \times H \times W }
$$
