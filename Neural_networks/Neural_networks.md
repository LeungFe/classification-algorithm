# Neural networks

感知机是二分类的线性分类模型，其输入是样本的特征向量，输出是样本的类别。感知机对应于输入空间中将样本划分为两个类别，属于判别模型。感知机学习旨在求出将训练数据进行线性划分的分离超平面，为此，导入基于误分类的损失函数，利用梯度下降法对损失函数进行极小化，求得感知机模型。

1、感知机模型

假设输入空间是<a href="https://www.codecogs.com/eqnedit.php?latex=X\subseteq&space;\mathbb{R}^{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X\subseteq&space;\mathbb{R}^{n}" title="X\subseteq \mathbb{R}^{n}" /></a>，输出空间是<a href="https://www.codecogs.com/eqnedit.php?latex=Y=\left&space;\{&space;&plus;1,-1&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y=\left&space;\{&space;&plus;1,-1&space;\right&space;\}" title="Y=\left \{ +1,-1 \right \}" /></a>。输入<a href="https://www.codecogs.com/eqnedit.php?latex=x\in&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x\in&space;X" title="x\in X" /></a>表示样本的特征向量，对应于输入空间的点，输出<a href="https://www.codecogs.com/eqnedit.php?latex=y\in&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y\in&space;Y" title="y\in Y" /></a>表示样本的类别。由输入空间到输出空间的函数如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)=sign(\omega\cdot&space;x&plus;b)\quad\quad(1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)=sign(\omega\cdot&space;x&plus;b)\quad\quad(1)" title="f(x)=sign(\omega\cdot x+b)\quad\quad(1)" /></a>

称为感知机，其中权值<a href="https://www.codecogs.com/eqnedit.php?latex=\omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega" title="\omega" /></a>和偏置b是感知机模型的参数。
感知机是一种线性分类模型，属于判别模型，感知机的假设空间定义在特征空间中所有线性分类模型。

2、多层感知机及其BP算法

Deep Learning近年来在各个领域都取得了state-of-the-art的效果，对于原始未加工且单独不可解释的特征尤为有效，传统的方法依赖手工选取特征，而 神经网络可以进行学习，通过层次结构学习到更利于任务的特征。得益于近年来互联网充足的数据，计算机硬件的发展以及大规模并行化的普及。本文主要简单介绍MLP，也即为Full-connection Neural Network ，网络结构如下，分为输入，隐层与输出层，除了输入层外，其余的每层激活函数均采用sigmod，MLP容易受到局部极小值与梯度弥散的困扰，如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019060219455683.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc2NjE3OQ==,size_16,color_FFFFFF,t_70)

MLP 的 BP 算法基于经典的链式求导法则，首先看前向传导，对于输入层有I个单元， 对于输入样本(x,z)，隐层的输入为：

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{h}=\sum_{i=1}^{I}\omega_{ih}x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{h}=\sum_{i=1}^{I}\omega_{ih}x_{i}" title="\alpha_{h}=\sum_{i=1}^{I}\omega_{ih}x_{i}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=b_{h}=f(\omega_{h})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_{h}=f(\omega_{h})" title="b_{h}=f(\omega_{h})" /></a>

这里函数f为非线性激活函数，常见的有sigmod或者是tanh，本文选取sigmod作为激活函数。计算完输入层向第一个隐层的传导后，剩下的隐层计算方式类似，用<a href="https://www.codecogs.com/eqnedit.php?latex=h_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{l}" title="h_{l}" /></a>表示第l层的单元数：

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{h}=\sum_{h^{'}=1}^{h_{l}-1}\omega_{h^{'}h}b_{h^{'}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{h}=\sum_{h^{'}=1}^{h_{l}-1}\omega_{h^{'}h}b_{h^{'}}" title="\alpha_{h}=\sum_{h^{'}=1}^{h_{l}-1}\omega_{h^{'}h}b_{h^{'}}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=b_{h}=f(\omega_{h})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_{h}=f(\omega_{h})" title="b_{h}=f(\omega_{h})" /></a>

对于输出层，若采用二分类即logistic regression ，则前向传导到输出层：

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha=\sum_{h^{'}}\omega_{h^{'}h}b_{h^{'}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha=\sum_{h^{'}}\omega_{h^{'}h}b_{h^{'}}" title="\alpha=\sum_{h^{'}}\omega_{h^{'}h}b_{h^{'}}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=y=f(\alpha)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=f(\alpha)" title="y=f(\alpha)" /></a>

这里 y 即为 MLP 的输出类别为1的概率，输出类别为0的概率为1−y，为了训练网络，当z=1时，y越大越好，而当z=0时，1−y越大越好 ，这样才能得到最优的参数w，采用MLE的方法，写到一起可以得到<a href="https://www.codecogs.com/eqnedit.php?latex=y^{z}(1-y)^{1-z}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^{z}(1-y)^{1-z}" title="y^{z}(1-y)^{1-z}" /></a>,这便是单个样本的似然函数，对于所有样本可以列出 log 似然函数<a href="https://www.codecogs.com/eqnedit.php?latex=O=∑_{x,z}zlogy&plus;(1−z)log(1−y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O=∑_{x,z}zlogy&plus;(1−z)log(1−y)" title="O=∑_{x,z}zlogy+(1−z)log(1−y)" /></a> ，直接极大化该似然函数即可，等价于极小化以下的−log损失函数：

<a href="https://www.codecogs.com/eqnedit.php?latex=O=-\left&space;[&space;\sum_{(x,z)}zlogy&plus;(1-z)log(1-y)&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O=-\left&space;[&space;\sum_{(x,z)}zlogy&plus;(1-z)log(1-y)&space;\right&space;]" title="O=-\left [ \sum_{(x,z)}zlogy+(1-z)log(1-y) \right ]" /></a>

对于多分类问题，即输出层采用softmax，假设有K个类别，则输出层的第k个单元计算过程如下:

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{k}=\sum_{h^{'}}\omega_{h^{'}k}b_{h^{'}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{k}=\sum_{h^{'}}\omega_{h^{'}k}b_{h^{'}}" title="\alpha_{k}=\sum_{h^{'}}\omega_{h^{'}k}b_{h^{'}}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=y_{k}=f(\alpha_{k})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{k}=f(\alpha_{k})" title="y_{k}=f(\alpha_{k})" /></a>

则得到类别k的概率可以写为$$ ，注意标签z中只有第k维为1，其余为0，所以现在只需极大化该似然函数即可:

<a href="https://www.codecogs.com/eqnedit.php?latex=O=\prod_{(x,z)}\prod_{k}y^{zk}_{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O=\prod_{(x,z)}\prod_{k}y^{zk}_{k}" title="O=\prod_{(x,z)}\prod_{k}y^{zk}_{k}" /></a>

同理等价于极小化以下损失：

<a href="https://www.codecogs.com/eqnedit.php?latex=O=-\prod_{(x,z)}\prod_{k}y^{zk}_{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O=-\prod_{(x,z)}\prod_{k}y^{zk}_{k}" title="O=-\prod_{(x,z)}\prod_{k}y^{zk}_{k}" /></a>

以上便是softmax的损失函数，这里需要注意的是以上优化目标O均没带正则项，而且logistic与softmax最后得到的损失函数均可以称作交叉熵损失，注意和平方损失的区别。

反向传播过程

有了以上前向传导的过程，接下来看误差的反向传递，对于sigmod来说，最后一层的计算如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha=\sum_{h}\omega_{h}\cdot&space;b_{h},y=f(\alpha)=\sigma&space;(\alpha)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha=\sum_{h}\omega_{h}\cdot&space;b_{h},y=f(\alpha)=\sigma&space;(\alpha)" title="\alpha=\sum_{h}\omega_{h}\cdot b_{h},y=f(\alpha)=\sigma (\alpha)" /></a>

这里<a href="https://www.codecogs.com/eqnedit.php?latex=b_{h}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_{h}" title="b_{h}" /></a>为倒数第二层单元h的输出，σ为sigmod激活函数，且满足σ′(a)=σ(a)(1−σ(a))，对于单个样本的损失 ：

<a href="https://www.codecogs.com/eqnedit.php?latex=O=-\left&space;[&space;zlog(\sigma&space;(\alpha))&plus;(1-z)log(1-\sigma&space;(\alpha))&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O=-\left&space;[&space;zlog(\sigma&space;(\alpha))&plus;(1-z)log(1-\sigma&space;(\alpha))&space;\right&space;]" title="O=-\left [ zlog(\sigma (\alpha))+(1-z)log(1-\sigma (\alpha)) \right ]" /></a>

可得到如下的链式求导过程：

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;O}{\partial&space;\omega_{h}}=\frac{\partial&space;O}{\partial&space;\alpha}\cdot&space;\frac{\partial&space;\alpha}{\partial\omega&space;_{h}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;O}{\partial&space;\omega_{h}}=\frac{\partial&space;O}{\partial&space;\alpha}\cdot&space;\frac{\partial&space;\alpha}{\partial\omega&space;_{h}}" title="\frac{\partial O}{\partial \omega_{h}}=\frac{\partial O}{\partial \alpha}\cdot \frac{\partial \alpha}{\partial\omega _{h}}" /></a>

显而易见对于后半部分<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\alpha}{\partial\omega&space;_{h}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\alpha}{\partial\omega&space;_{h}}" title="\frac{\partial \alpha}{\partial\omega _{h}}" /></a>为$b_{h}$，对于前半部分$\frac{\partial O}{\partial \alpha}$：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602201458449.png)

以上，便得到了logistic的残差，接下来残差反向传递即可，残差传递形式同softmax，所以先推导softmax的残差项，对于单个样本，softmax的log损失函数为：

<a href="https://www.codecogs.com/eqnedit.php?latex=O=-\sum&space;_{i}z_{i}logy_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O=-\sum&space;_{i}z_{i}logy_{i}" title="O=-\sum _{i}z_{i}logy_{i}" /></a>

其中：

<a href="https://www.codecogs.com/eqnedit.php?latex=y_{i}=\frac{e^{\alpha_{i}}}{\sum&space;_{j}e^{\alpha_{j}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{i}=\frac{e^{\alpha_{i}}}{\sum&space;_{j}e^{\alpha_{j}}}" title="y_{i}=\frac{e^{\alpha_{i}}}{\sum _{j}e^{\alpha_{j}}}" /></a>

根据以上分析，可得到<a href="https://www.codecogs.com/eqnedit.php?latex=y_{k^{'}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{k^{'}}" title="y_{k^{'}}" /></a>关于<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{k}" title="\alpha_{k}" /></a>的导数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602201912319.png)

现在能得到损失函数O对于<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{k}" title="\alpha_{k}" /></a>的导数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602202002159.png)

这里有<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i}z_{i}=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i}z_{i}=1" title="\sum_{i}z_{i}=1" /></a>，即只有一个类别，到这一步，softmax和sigmod的残差均计算完成，可用<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a>来表示，对于单元j，其形式如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_{j}=\frac{\partial&space;O}{\partial&space;\alpha_{j}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_{j}=\frac{\partial&space;O}{\partial&space;\alpha_{j}}" title="\sigma_{j}=\frac{\partial O}{\partial \alpha_{j}}" /></a>

这里可以得到softmax层向倒数第二层的残差反向传递公式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602202520987.png)

其中<a href="https://www.codecogs.com/eqnedit.php?latex=a_{k}=\sum_{h}w_{hk}b_{h}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_{k}=\sum_{h}w_{hk}b_{h}" title="a_{k}=\sum_{h}w_{hk}b_{h}" /></a> ，对于sigmod层，向倒数第二层的反向传递公式为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602202642673.png)

以上公式的 δ 代表sigmod层唯一的残差，接下来就是残差从隐层向前传递的传递过程，一直传递到首个隐藏层即第二层（注意，残差不会传到输入层，因为不需要，对输入层到第二层的参数求导，其只依赖于第二层的残差，因为第二层是这些参数的放射函数）：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602202722337.png)

整个过程可以看下图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602202827896.gif)

最终得到关于权值的计算公式：

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;O}{\partial&space;\omega_{ij}}=\frac{\partial&space;O}{\partial&space;\alpha_{j}}\cdot&space;\frac{\partial&space;\alpha_{j}}{\partial&space;\omega_{ij}}=\sigma_{j}b_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;O}{\partial&space;\omega_{ij}}=\frac{\partial&space;O}{\partial&space;\alpha_{j}}\cdot&space;\frac{\partial&space;\alpha_{j}}{\partial&space;\omega_{ij}}=\sigma_{j}b_{i}" title="\frac{\partial O}{\partial \omega_{ij}}=\frac{\partial O}{\partial \alpha_{j}}\cdot \frac{\partial \alpha_{j}}{\partial \omega_{ij}}=\sigma_{j}b_{i}" /></a>

至此完成了backwark pass的过程，注意由于计算比较复杂，有必要进行梯度验证。对函数O关于参数<a href="https://www.codecogs.com/eqnedit.php?latex=\omega_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega_{ij}" title="\omega_{ij}" /></a>进行数值求导即可，求导之后与与上边的公式验证差异，小于给定的阈值即认为我们的运算是正确的。

