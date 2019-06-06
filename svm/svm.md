# Support-Vector-Machine

训练集：<a href="https://www.codecogs.com/eqnedit.php?latex=X={(x_{1},y_{1}),(x_{2},y_{2}),...,(x_{m},y_{m})},y_{i}\in\left&space;\{&space;-1,1&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X={(x_{1},y_{1}),(x_{2},y_{2}),...,(x_{m},y_{m})},y_{i}\in\left&space;\{&space;-1,1&space;\right&space;\}" title="X={(x_{1},y_{1}),(x_{2},y_{2}),...,(x_{m},y_{m})},y_{i}\in\left \{ -1,1 \right \}" /></a>。SVM分类器就是要找到一个超平面，把不同类别的样本分开，但是这样的超平面有很多，如下图所示，那么到底该选择哪一个？

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526110416537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc2NjE3OQ==,size_16,color_FFFFFF,t_70)

直观上看，应该选择红色的超平面，因为这个超平面使得训练样本的容错能力最强，对测试集的泛化能力最强。

样本空间任意点x到超平面的距离为

<a href="https://www.codecogs.com/eqnedit.php?latex=r=\frac{|\omega&space;^{T}x&plus;b|}{\left&space;\|&space;\omega&space;\right&space;\|}\left&space;(&space;1.1&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r=\frac{|\omega&space;^{T}x&plus;b|}{\left&space;\|&space;\omega&space;\right&space;\|}\left&space;(&space;1.1&space;\right&space;)" title="r=\frac{|\omega ^{T}x+b|}{\left \| \omega \right \|}\left ( 1.1 \right )" /></a>

假设超平面能将训练样本正确分类：

<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;&\omega&space;^{T}x_{i}&plus;b\geq&space;&plus;1,y_{i}=&plus;1&space;\\&space;&\omega&space;^{T}x_{i}&plus;b\leq&space;-1,y_{i}=-1&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;&\omega&space;^{T}x_{i}&plus;b\geq&space;&plus;1,y_{i}=&plus;1&space;\\&space;&\omega&space;^{T}x_{i}&plus;b\leq&space;-1,y_{i}=-1&space;\end{matrix}\right." title="\left\{\begin{matrix} &\omega ^{T}x_{i}+b\geq +1,y_{i}=+1 \\ &\omega ^{T}x_{i}+b\leq -1,y_{i}=-1 \end{matrix}\right." /></a>（1.2）

如下图所示，距离超平面最近的样本点能够使得（1.2）的等号成立，这些样本点称为支持向量，所以两个不同类别的支持向量到超平面的距离之和（称为间隔）为

<a href="https://www.codecogs.com/eqnedit.php?latex=\gamma&space;=\frac{2}{\left&space;\|&space;\omega&space;\right&space;\|}\left&space;(&space;1.3&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma&space;=\frac{2}{\left&space;\|&space;\omega&space;\right&space;\|}\left&space;(&space;1.3&space;\right&space;)" title="\gamma =\frac{2}{\left \| \omega \right \|}\left ( 1.3 \right )" /></a>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526110416537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc2NjE3OQ==,size_16,color_FFFFFF,t_70)

要找到最大间隔的超平面，就是要找到满足（1.2）中约束的参数w和b，使得间隔最大，即

<a href="https://www.codecogs.com/eqnedit.php?latex=\max_{\omega&space;,b}\frac{2}{\left&space;\|&space;\omega&space;\right&space;\|}\\&space;s.t.y_{i}(\omega&space;^{T}x_{i}&plus;b)\geq&space;1,i=1,2,...,m(1.4)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max_{\omega&space;,b}\frac{2}{\left&space;\|&space;\omega&space;\right&space;\|}\\&space;s.t.y_{i}(\omega&space;^{T}x_{i}&plus;b)\geq&space;1,i=1,2,...,m(1.4)" title="\max_{\omega ,b}\frac{2}{\left \| \omega \right \|}\\ s.t.y_{i}(\omega ^{T}x_{i}+b)\geq 1,i=1,2,...,m(1.4)" /></a>

最大化<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{2}{||\omega||}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{2}{||\omega||}" title="\frac{2}{||\omega||}" /></a>，等价于最小化<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{||\omega||}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{||\omega||}{2}" title="\frac{||\omega||}{2}" /></a>，也就是等价于最小化<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{||\omega||^{2&space;}}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{||\omega||^{2&space;}}{2}" title="\frac{||\omega||^{2 }}{2}" /></a>，所以

<a href="https://www.codecogs.com/eqnedit.php?latex=\min_{\omega&space;,b}\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}\\&space;s.t.y_{i}(\omega&space;^{T}x_{i}&plus;b)\geq&space;1,i=1,2,...,m(1.5)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{\omega&space;,b}\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}\\&space;s.t.y_{i}(\omega&space;^{T}x_{i}&plus;b)\geq&space;1,i=1,2,...,m(1.5)" title="\min_{\omega ,b}\frac{1}{2}\left \| \omega \right \|^{2}\\ s.t.y_{i}(\omega ^{T}x_{i}+b)\geq 1,i=1,2,...,m(1.5)" /></a>

**1.2对偶问题**

对公式（1.5）使用拉格朗日乘子法得到它的对偶问题，具体来说，对（1.5）的每条约束添加拉格朗日乘子<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{i}\geq0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{i}\geq0" title="\alpha_{i}\geq0" /></a>，则该问题的拉格朗日函数可写为

<a href="https://www.codecogs.com/eqnedit.php?latex=L(\omega&space;,b,\alpha&space;)=\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}&plus;\sum_{i=1}^{m}\alpha&space;_{i}(1-y_{i}(\omega&space;^{T}x_{i}&plus;b))\left&space;(&space;2.1&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\omega&space;,b,\alpha&space;)=\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}&plus;\sum_{i=1}^{m}\alpha&space;_{i}(1-y_{i}(\omega&space;^{T}x_{i}&plus;b))\left&space;(&space;2.1&space;\right&space;)" title="L(\omega ,b,\alpha )=\frac{1}{2}\left \| \omega \right \|^{2}+\sum_{i=1}^{m}\alpha _{i}(1-y_{i}(\omega ^{T}x_{i}+b))\left ( 2.1 \right )" /></a>

其中<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha=(\alpha_{1},\alpha_{2},...,\alpha_{m})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha=(\alpha_{1},\alpha_{2},...,\alpha_{m})" title="\alpha=(\alpha_{1},\alpha_{2},...,\alpha_{m})" /></a>，令<a href="https://www.codecogs.com/eqnedit.php?latex=L(\omega,b,\alpha)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\omega,b,\alpha)" title="L(\omega,b,\alpha)" /></a>对<a href="https://www.codecogs.com/eqnedit.php?latex=\omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega" title="\omega" /></a>和b的偏导为0，得

<a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;=\sum_{i=1}^{m}\alpha&space;_{i}y_{i}x_{i}(2.2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega&space;=\sum_{i=1}^{m}\alpha&space;_{i}y_{i}x_{i}(2.2)" title="\omega =\sum_{i=1}^{m}\alpha _{i}y_{i}x_{i}(2.2)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i=1}^{m}\alpha&space;_{i}y_{i}=0\left&space;(&space;2.3&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{m}\alpha&space;_{i}y_{i}=0\left&space;(&space;2.3&space;\right&space;)" title="\sum_{i=1}^{m}\alpha _{i}y_{i}=0\left ( 2.3 \right )" /></a>

把（2.2）和（2.3）代入（2.1），整理得到对偶问题

<a href="https://www.codecogs.com/eqnedit.php?latex=\max&space;_{\alpha}\sum_{i=1}^{m}\alpha&space;_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha&space;_{i}\alpha&space;_{j}y_{i}y_{j}x_{i}^{T}x_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max&space;_{\alpha}\sum_{i=1}^{m}\alpha&space;_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha&space;_{i}\alpha&space;_{j}y_{i}y_{j}x_{i}^{T}x_{j}" title="\max _{\alpha}\sum_{i=1}^{m}\alpha _{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha _{i}\alpha _{j}y_{i}y_{j}x_{i}^{T}x_{j}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=s.t.\sum_{i=1}^{m}=0,\alpha_{i}\geq&space;0,i=1,2,...,m(2.4)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s.t.\sum_{i=1}^{m}=0,\alpha_{i}\geq&space;0,i=1,2,...,m(2.4)" title="s.t.\sum_{i=1}^{m}=0,\alpha_{i}\geq 0,i=1,2,...,m(2.4)" /></a>

解出<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a>，求出<a href="https://www.codecogs.com/eqnedit.php?latex=\omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega" title="\omega" /></a>和b即可得到模型：

<a href="https://www.codecogs.com/eqnedit.php?latex=f=\omega^{T}x&plus;b=\sum_{i=1}^{m}\alpha_{i}y_{i}x_{i}^{T}x&plus;b\left&space;(&space;2.5&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f=\omega^{T}x&plus;b=\sum_{i=1}^{m}\alpha_{i}y_{i}x_{i}^{T}x&plus;b\left&space;(&space;2.5&space;\right&space;)" title="f=\omega^{T}x+b=\sum_{i=1}^{m}\alpha_{i}y_{i}x_{i}^{T}x+b\quad\quad \eqno{(2.5)}" /></a>

上述过程需要满足的KKT条件：

<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;&\alpha_{i}\geq&space;0&space;\\&space;&y_{i}f(x_{i})-1\geq&space;0&space;\\&space;&\alpha_{i}(y_{i}f(x_{i})-1)=0&space;\end{matrix}\right.(2.6)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;&\alpha_{i}\geq&space;0&space;\\&space;&y_{i}f(x_{i})-1\geq&space;0&space;\\&space;&\alpha_{i}(y_{i}f(x_{i})-1)=0&space;\end{matrix}\right.(2.6)" title="\left\{\begin{matrix} &\alpha_{i}\geq 0 \\ &y_{i}f(x_{i})-1\geq 0 \\ &\alpha_{i}(y_{i}f(x_{i})-1)=0 \end{matrix}\right.\quad\quad \eqno{(2.6)}" /></a>

**1.3核函数**

之前讨论的情况都是线性可分的，但是在实际的样本空间中并不存在能够正确划分两个类别的线性分类器，如下图所示的异或问题就不是线性可分的。对于这样的问题，可以将原始的空间映射到一个高维的特征空间，使得样本在高维特征空间线性可分。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526153522981.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc2NjE3OQ==,size_16,color_FFFFFF,t_70)

令<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(x)" title="\phi(x)" /></a>表示将x映射后的特征向量，于是，在特征空间划分超平面所对应的模型可以表示为

<a href="https://www.codecogs.com/eqnedit.php?latex=f=\omega^{T}\phi(x)&plus;b\quad\quad&space;\eqno{(3.1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f=\omega^{T}\phi(x)&plus;b\quad\quad&space;\eqno{(3.1)}" title="f=\omega^{T}\phi(x)+b\quad\quad \eqno{(3.1)}" /></a>

其中，<a href="https://www.codecogs.com/eqnedit.php?latex=\omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega" title="\omega" /></a>和b是模型参数。原问题：

<a href="https://www.codecogs.com/eqnedit.php?latex=\min&space;_{\omega,b}\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}\\&space;s.t.\quad&space;y_{i}(\omega^{T}\phi(x_{i})&plus;b)\geq&space;1,&space;i=1,2,...,m\quad\quad(3.2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min&space;_{\omega,b}\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}\\&space;s.t.\quad&space;y_{i}(\omega^{T}\phi(x_{i})&plus;b)\geq&space;1,&space;i=1,2,...,m\quad\quad(3.2)" title="\min _{\omega,b}\frac{1}{2}\left \| \omega \right \|^{2}\\ s.t.\quad y_{i}(\omega^{T}\phi(x_{i})+b)\geq 1, i=1,2,...,m\quad\quad(3.2)" /></a>

对偶问题：

<a href="https://www.codecogs.com/eqnedit.php?latex=\max&space;_{\omega}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_{i})^{T}\phi(x_{j})\\&space;s.t.&space;\quad&space;\sum_{i=1}^{m}\alpha_{i}y_{i}=&space;0,\alpha_{i}\geq&space;0,i=1,2,...,m.\quad\quad(3.3)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max&space;_{\omega}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_{i})^{T}\phi(x_{j})\\&space;s.t.&space;\quad&space;\sum_{i=1}^{m}\alpha_{i}y_{i}=&space;0,\alpha_{i}\geq&space;0,i=1,2,...,m.\quad\quad(3.3)" title="\max _{\omega}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_{i})^{T}\phi(x_{j})\\ s.t. \quad \sum_{i=1}^{m}\alpha_{i}y_{i}= 0,\alpha_{i}\geq 0,i=1,2,...,m.\quad\quad(3.3)" /></a>

其中<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(x_{i})^{T}\phi(x_{j})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(x_{i})^{T}\phi(x_{j})" title="\phi(x_{i})^{T}\phi(x_{j})" /></a>是<a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{i}" title="x_{i}" /></a>与<a href="https://www.codecogs.com/eqnedit.php?latex=x_{}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{}" title="x_{}" /></a>映射到特征空间之后的内积，由于特征空间维数比较高，甚至无穷维，因此直接计算<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(x_{i})^{T}\phi(x_{j})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(x_{i})^{T}\phi(x_{j})" title="\phi(x_{i})^{T}\phi(x_{j})" /></a>通常非常困难，所以，引入核函数来解决这个问题。<a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{i}" title="x_{i}" /></a>与<a href="https://www.codecogs.com/eqnedit.php?latex=x_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{j}" title="x_{j}" /></a>在特征空间的内积等于它们在原始空间中通过核函数计算的结果，这样就可以避免在高维甚至无穷维特征空间中做内积，减少计算的复杂度。

<a href="https://www.codecogs.com/eqnedit.php?latex=k(x_{i},x_{j})=\left&space;\langle&space;\phi(x_{i}),\phi(x_{j})&space;\right&space;\rangle=\phi(x_{i})^{T}\phi(x_{j})\quad\quad(3.4)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k(x_{i},x_{j})=\left&space;\langle&space;\phi(x_{i}),\phi(x_{j})&space;\right&space;\rangle=\phi(x_{i})^{T}\phi(x_{j})\quad\quad(3.4)" title="k(x_{i},x_{j})=\left \langle \phi(x_{i}),\phi(x_{j}) \right \rangle=\phi(x_{i})^{T}\phi(x_{j})\quad\quad(3.4)" /></a>

公式（3.3）可以写成

<a href="https://www.codecogs.com/eqnedit.php?latex=\max&space;_{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}k(x_{i},x_{j})\\&space;s.t.\quad&space;\sum_{i=1}^{m}\alpha_{i}y_{i}=0,\alpha_{i}\geq&space;0,i=1,2,...,m.\quad\quad(3.5)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max&space;_{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}k(x_{i},x_{j})\\&space;s.t.\quad&space;\sum_{i=1}^{m}\alpha_{i}y_{i}=0,\alpha_{i}\geq&space;0,i=1,2,...,m.\quad\quad(3.5)" title="\max _{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}k(x_{i},x_{j})\\ s.t.\quad \sum_{i=1}^{m}\alpha_{i}y_{i}=0,\alpha_{i}\geq 0,i=1,2,...,m.\quad\quad(3.5)" /></a>

求解后得到：

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)=\omega^{T}\phi(x)&plus;b\\&space;=\sum_{i=1}^{m}\alpha_{i}y_{i}\phi(x_{i})^{T}\phi(x)&plus;b\\&space;=\sum_{i=1}^{m}\alpha_{i}y_{i}k(x,x_{i})&plus;b\quad\quad(3.6)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)=\omega^{T}\phi(x)&plus;b\\&space;=\sum_{i=1}^{m}\alpha_{i}y_{i}\phi(x_{i})^{T}\phi(x)&plus;b\\&space;=\sum_{i=1}^{m}\alpha_{i}y_{i}k(x,x_{i})&plus;b\quad\quad(3.6)" title="f(x)=\omega^{T}\phi(x)+b\\ =\sum_{i=1}^{m}\alpha_{i}y_{i}\phi(x_{i})^{T}\phi(x)+b\\ =\sum_{i=1}^{m}\alpha_{i}y_{i}k(x,x_{i})+b\quad\quad(3.6)" /></a>

常见的核函数：

线性核：

<a href="https://www.codecogs.com/eqnedit.php?latex=k(x_{i},x_{j})=x_{i}^{T}x_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k(x_{i},x_{j})=x_{i}^{T}x_{j}" title="k(x_{i},x_{j})=x_{i}^{T}x_{j}" /></a>

高斯核：

<a href="https://www.codecogs.com/eqnedit.php?latex=k(x_{i},x_{j})=exp(-\frac{\left&space;\|&space;x_{i}-x_{j}&space;\right&space;\|^{2}}{2\sigma&space;^{2}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k(x_{i},x_{j})=exp(-\frac{\left&space;\|&space;x_{i}-x_{j}&space;\right&space;\|^{2}}{2\sigma&space;^{2}})" title="k(x_{i},x_{j})=exp(-\frac{\left \| x_{i}-x_{j} \right \|^{2}}{2\sigma ^{2}})" /></a>

多项式核：

<a href="https://www.codecogs.com/eqnedit.php?latex=k(x_{i},x_{j})=(x_{i}^{T}x_{j})^{d}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k(x_{i},x_{j})=(x_{i}^{T}x_{j})^{d}" title="k(x_{i},x_{j})=(x_{i}^{T}x_{j})^{d}" /></a>

1.4软间隔与正则化

在之前的讨论中，都是假设训练样本在原始空间或者特征空间中是线性可分，但是实际中，大部分样本在原始空间或者特征空间中是不可分的，为了解决这个问题，支持向量机允许一些样本分类错误，也就是允许一些样本不满足<a href="https://www.codecogs.com/eqnedit.php?latex=y_{i}(\omega^{T}x_{i}&plus;b)\geq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{i}(\omega^{T}x_{i}&plus;b)\geq&space;1" title="y_{i}(\omega^{T}x_{i}+b)\geq 1" /></a>。为此，引入软间隔的概念，红色的点表示支持向量，如下图所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526204724867.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc2NjE3OQ==,size_16,color_FFFFFF,t_70)

当然在最大化间隔的同时，也要让不满足约束的样本尽可能少，所以，优化目标方程为

<a href="https://www.codecogs.com/eqnedit.php?latex=\min&space;_{\omega,b}\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}&plus;&space;C\sum_{i=1}^{m}\max&space;(0,1-y_{i}(\omega^{T}x_{i}&plus;b))\quad\quad(4.1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min&space;_{\omega,b}\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}&plus;&space;C\sum_{i=1}^{m}\max&space;(0,1-y_{i}(\omega^{T}x_{i}&plus;b))\quad\quad(4.1)" title="\min _{\omega,b}\frac{1}{2}\left \| \omega \right \|^{2}+ C\sum_{i=1}^{m}\max (0,1-y_{i}(\omega^{T}x_{i}+b))\quad\quad(4.1)" /></a>

引入松弛变量<a href="https://www.codecogs.com/eqnedit.php?latex=\xi_{i}\geq&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\xi_{i}\geq&space;0" title="\xi_{i}\geq 0" /></a>，将（4.1）写成

<a href="https://www.codecogs.com/eqnedit.php?latex=\min&space;_{\omega,b,\xi_{i}}\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}&plus;C\sum_{i=1}^{m}\xi_{i}\quad\quad(4.2)\\&space;s.t.\quad&space;y_{i}(\omega^{T}x_{i}&plus;b)\geq&space;1-\xi_{i},\xi_{i}\geq&space;0,i=1,2,...,m." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min&space;_{\omega,b,\xi_{i}}\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}&plus;C\sum_{i=1}^{m}\xi_{i}\quad\quad(4.2)\\&space;s.t.\quad&space;y_{i}(\omega^{T}x_{i}&plus;b)\geq&space;1-\xi_{i},\xi_{i}\geq&space;0,i=1,2,...,m." title="\min _{\omega,b,\xi_{i}}\frac{1}{2}\left \| \omega \right \|^{2}+C\sum_{i=1}^{m}\xi_{i}\quad\quad(4.2)\\ s.t.\quad y_{i}(\omega^{T}x_{i}+b)\geq 1-\xi_{i},\xi_{i}\geq 0,i=1,2,...,m." /></a>

拉格朗日方程为

<a href="https://www.codecogs.com/eqnedit.php?latex=L(\omega,b,\alpha,\xi,\mu&space;)=\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}&plus;C\sum_{i=1}^{m}\xi_{i}\\&space;&plus;\sum_{i=1}^{m}\alpha_{i}(1-\xi_{i}-y_{i}(\omega^{T}x_{i}&plus;b))-\sum_{i=1}^{m}\mu_{i}\xi_{i}\quad\quad(4.3)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\omega,b,\alpha,\xi,\mu&space;)=\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^{2}&plus;C\sum_{i=1}^{m}\xi_{i}\\&space;&plus;\sum_{i=1}^{m}\alpha_{i}(1-\xi_{i}-y_{i}(\omega^{T}x_{i}&plus;b))-\sum_{i=1}^{m}\mu_{i}\xi_{i}\quad\quad(4.3)" title="L(\omega,b,\alpha,\xi,\mu )=\frac{1}{2}\left \| \omega \right \|^{2}+C\sum_{i=1}^{m}\xi_{i}\\ +\sum_{i=1}^{m}\alpha_{i}(1-\xi_{i}-y_{i}(\omega^{T}x_{i}+b))-\sum_{i=1}^{m}\mu_{i}\xi_{i}\quad\quad(4.3)" /></a>

其中，<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{i}\geq&space;0,\mu_{i}\geq&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{i}\geq&space;0,\mu_{i}\geq&space;0" title="\alpha_{i}\geq 0,\mu_{i}\geq 0" /></a>是拉格朗日乘子。令拉格朗日函数对<a href="https://www.codecogs.com/eqnedit.php?latex=\omega,b,\xi_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega,b,\xi_{i}" title="\omega,b,\xi_{i}" /></a>的偏导为0，得

<a href="https://www.codecogs.com/eqnedit.php?latex=\omega=\sum_{i=1}^{m}\alpha_{i}y_{i}x_{i}\qued\quad(4.4)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega=\sum_{i=1}^{m}\alpha_{i}y_{i}x_{i}\qued\quad(4.4)" title="\omega=\sum_{i=1}^{m}\alpha_{i}y_{i}x_{i}\qued\quad(4.4)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i=1}^{m}\alpha_{i}y_{i}=0\qued\quad(4.5)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{m}\alpha_{i}y_{i}=0\qued\quad(4.5)" title="\sum_{i=1}^{m}\alpha_{i}y_{i}=0\qued\quad(4.5)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{i}&plus;\mu_{i}=C\qued\quad(4.6)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{i}&plus;\mu_{i}=C\qued\quad(4.6)" title="\alpha_{i}+\mu_{i}=C\qued\quad(4.6)" /></a>

将（4.4）-（4.6）带入（4.3）得到对偶问题：

<a href="https://www.codecogs.com/eqnedit.php?latex=\max&space;_{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}\quad\quad(4.7)\\&space;s.t.\quad&space;\sum_{i=1}^{m}\alpha_{i}y_{i}=0,0\leq&space;\alpha_{i}\leq&space;C,i=1,2,...,m." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max&space;_{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}\quad\quad(4.7)\\&space;s.t.\quad&space;\sum_{i=1}^{m}\alpha_{i}y_{i}=0,0\leq&space;\alpha_{i}\leq&space;C,i=1,2,...,m." title="\max _{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}\quad\quad(4.7)\\ s.t.\quad \sum_{i=1}^{m}\alpha_{i}y_{i}=0,0\leq \alpha_{i}\leq C,i=1,2,...,m." /></a>

同样，也可以引入核函数，得

<a href="https://www.codecogs.com/eqnedit.php?latex=\max&space;_{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_{i},x_{j})&space;\quad\quad(4.8)\\&space;s.t.\quad&space;\sum_{i=1}^{m}\alpha_{i}y_{i}=0,0\leq&space;\alpha_{i}\leq&space;C,i=1,2,...,m." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max&space;_{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_{i},x_{j})&space;\quad\quad(4.8)\\&space;s.t.\quad&space;\sum_{i=1}^{m}\alpha_{i}y_{i}=0,0\leq&space;\alpha_{i}\leq&space;C,i=1,2,...,m." title="\max _{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_{i},x_{j}) \quad\quad(4.8)\\ s.t.\quad \sum_{i=1}^{m}\alpha_{i}y_{i}=0,0\leq \alpha_{i}\leq C,i=1,2,...,m." /></a>

对于软间隔支持向量机，KKT条件要求

<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;\alpha_{i}\geq&space;0,\mu_{i}\geq&space;0&space;&&space;\\&space;y_{i}f(x_{i})-1&plus;\xi_{i}\geq&space;0&&space;\\&space;\alpha_{i}(y_{i}f(x_{i})-1&plus;\xi_{i})=0&&space;\\&space;\xi_{i}\geq&space;0,&space;\mu_{i}\xi_{i}=0&&space;\end{matrix}\right.\qued\qued(4.9)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;\alpha_{i}\geq&space;0,\mu_{i}\geq&space;0&space;&&space;\\&space;y_{i}f(x_{i})-1&plus;\xi_{i}\geq&space;0&&space;\\&space;\alpha_{i}(y_{i}f(x_{i})-1&plus;\xi_{i})=0&&space;\\&space;\xi_{i}\geq&space;0,&space;\mu_{i}\xi_{i}=0&&space;\end{matrix}\right.\qued\qued(4.9)" title="\left\{\begin{matrix} \alpha_{i}\geq 0,\mu_{i}\geq 0 & \\ y_{i}f(x_{i})-1+\xi_{i}\geq 0& \\ \alpha_{i}(y_{i}f(x_{i})-1+\xi_{i})=0& \\ \xi_{i}\geq 0, \mu_{i}\xi_{i}=0& \end{matrix}\right.\qued\qued(4.9)" /></a>

