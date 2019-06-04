# Neural networks

感知机是二分类的线性分类模型，其输入是样本的特征向量，输出是样本的类别。感知机对应于输入空间中将样本划分为两个类别，属于判别模型。感知机学习旨在求出将训练数据进行线性划分的分离超平面，为此，导入基于误分类的损失函数，利用梯度下降法对损失函数进行极小化，求得感知机模型。

1、感知机模型

假设输入空间是<a href="https://www.codecogs.com/eqnedit.php?latex=X\subseteq&space;\mathbb{R}^{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X\subseteq&space;\mathbb{R}^{n}" title="X\subseteq \mathbb{R}^{n}" /></a>，输出空间是<a href="https://www.codecogs.com/eqnedit.php?latex=Y=\left&space;\{&space;&plus;1,-1&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y=\left&space;\{&space;&plus;1,-1&space;\right&space;\}" title="Y=\left \{ +1,-1 \right \}" /></a>。输入<a href="https://www.codecogs.com/eqnedit.php?latex=x\in&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x\in&space;X" title="x\in X" /></a>表示样本的特征向量，对应于输入空间的点，输出<a href="https://www.codecogs.com/eqnedit.php?latex=y\in&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y\in&space;Y" title="y\in Y" /></a>表示样本的类别。由输入空间到输出空间的函数如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)=sign(\omega\cdot&space;x&plus;b)\quad\quad(1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)=sign(\omega\cdot&space;x&plus;b)\quad\quad(1)" title="f(x)=sign(\omega\cdot x+b)\quad\quad(1)" /></a>

称为感知机，其中权值$\omega$和偏置b是感知机模型的参数。
感知机是一种线性分类模型，属于判别模型，感知机的假设空间定义在特征空间中所有线性分类模型。
