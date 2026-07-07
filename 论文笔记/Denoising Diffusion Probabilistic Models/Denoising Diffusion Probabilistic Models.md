$x_0\sim q(x_0)$来源于数据集
$p _ { \theta } ( { \bf x } _ { 0 : T } )$称为反向过程
$$
p _ { \theta } ( { \bf x } _ { 0 : T } ):= p ( { \bf x } _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( { \bf x } _ { t - 1 } | { \bf x } _ { t } ), \qquad p _ { \theta } ( { \bf x } _ { t - 1 } | { \bf x } _ { t } ) := { \cal N } ( { \bf x } _ { t - 1 } ; \boldsymbol { \mu } _ { \theta } ( { \bf x } _ { t }, t ), \boldsymbol { \sum } _ { \theta } ( { \bf x } _ { t }, t ) )
$$
注意第一个式子，他定义了$p_\theta$的Markov性
