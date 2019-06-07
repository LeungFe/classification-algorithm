
# naive_bayes

正向概率：假设袋子里面有N个红球，M个黑球，取出一个球，此时摸出黑球的概率是多少。

逆向概率：事先不知道袋子里面红球和黑球数量的比例，取出一个球或者多个球，观察这些取出的球的颜色，可以对袋子中红球和黑球的比例做出推断。

朴素贝叶斯是为了解决逆向概率问题。

朴素贝叶斯算法和之前介绍的分类算法不同，之前介绍的分类算法都是判别模型，如支持向量机、K近邻、决策树等等。判别模型是直接学习出输出Y和特征X之间的关系，直接学习决策函数Y=f(X)，或者条件分布P(Y|X)。而朴素贝叶斯算法是生成模型，它是直接找出输出Y和特征X的联合分布，然后利用<a href="https://www.codecogs.com/eqnedit.php?latex=P(Y|X)=\frac{P(X,Y)}{P(X)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y|X)=\frac{P(X,Y)}{P(X)}" title="P(Y|X)=\frac{P(X,Y)}{P(X)}" /></a>得出。

先看看条件独立公式，如果X和Y是相互独立的，那么：

<a href="https://www.codecogs.com/eqnedit.php?latex=P(X,Y)=P(X)P(Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(X,Y)=P(X)P(Y)" title="P(X,Y)=P(X)P(Y)" /></a>

条件概率公式：

<a href="https://www.codecogs.com/eqnedit.php?latex=P(Y|X)=\frac{P(X|Y)P(Y)}{P(X)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y|X)=\frac{P(X|Y)P(Y)}{P(X)}" title="P(Y|X)=\frac{P(X|Y)P(Y)}{P(X)}" /></a>

全概率公式：

<a href="https://www.codecogs.com/eqnedit.php?latex=P(X)=\sum_{k=1}^{m}P(X|Y=Y_{k})P(X)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(X)=\sum_{k=1}^{m}P(X|Y=Y_{k})P(X)" title="P(X)=\sum_{k=1}^{m}P(X|Y=Y_{k})P(X)" /></a>

贝叶斯公式：

<a href="https://www.codecogs.com/eqnedit.php?latex=P(Y_{k}|X)=\frac{P(X|Y_{k})P(Y_{k})}{\sum_{k=1}^{m}P(X|Y=Y_{k})P(Y_{k})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y_{k}|X)=\frac{P(X|Y_{k})P(Y_{k})}{\sum_{k=1}^{m}P(X|Y=Y_{k})P(Y_{k})}" title="P(Y_{k}|X)=\frac{P(X|Y_{k})P(Y_{k})}{\sum_{k=1}^{m}P(X|Y=Y_{k})P(Y_{k})}" /></a>

朴素贝叶斯的假设：每个特征都是独立同分布的，每个特征同等重要，即每个特征是等权重的，只考虑特征是否出现，不考虑特征出现的次数。

以垃圾邮件分类为例子来介绍朴素贝叶斯，假设邮件$X=(x_{1},x_{2},...,x_{n})$，$y=0$表示正常邮件，$y=1$表示垃圾邮件，判断该邮件是否为垃圾邮件，只需要判断$P(y=0|X)$和$P(y=1|X)$哪个概率大就可以。

<a href="https://www.codecogs.com/eqnedit.php?latex=P(y=0|X)=\frac{P(X|y=0)*P(y=0)}{P(X)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(y=0|X)=\frac{P(X|y=0)*P(y=0)}{P(X)}" title="P(y=0|X)=\frac{P(X|y=0)*P(y=0)}{P(X)}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(y=1|X)=\frac{P(X|y=1)*P(y=1)}{P(X)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(y=1|X)=\frac{P(X|y=1)*P(y=1)}{P(X)}" title="P(y=1|X)=\frac{P(X|y=1)*P(y=1)}{P(X)}" /></a>

其中$P(y=0)$和$P(y=1)$表示先验概率，$P(X|y=0)$和$P(X|y=1)$表示条件概率。

<a href="https://www.codecogs.com/eqnedit.php?latex=P(y=0)=\frac{N_{y=0}}{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(y=0)=\frac{N_{y=0}}{N}" title="P(y=0)=\frac{N_{y=0}}{N}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(y=1)=\frac{N_{y=1}}{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(y=1)=\frac{N_{y=1}}{N}" title="P(y=1)=\frac{N_{y=1}}{N}" /></a>

其中$N_{y=0}$表示正常邮件的数量，$N_{y=1}$表示垃圾邮件的数量，$N$表示总邮件数量。

$P(X|y=0)$可以表示为$P(x_{1},x_{2},...,x_{n}|y=0)$，$P(X|y=1)$可以表示为$P(x_{1},x_{2},...,x_{n}|y=1)$，因为朴素贝叶斯假设每个特征是相互独立的，所以：

<a href="https://www.codecogs.com/eqnedit.php?latex=P(x_{1},x_{2},...,x_{n}|y=0)=P(x_{1}|y=0)\cdot&space;P(x_{2}|y=0)\cdot&space;...\cdot&space;P(x_{n}|y=0)\\&space;=\prod_{k=1}^{n}P(x_{k}|y=0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x_{1},x_{2},...,x_{n}|y=0)=P(x_{1}|y=0)\cdot&space;P(x_{2}|y=0)\cdot&space;...\cdot&space;P(x_{n}|y=0)\\&space;=\prod_{k=1}^{n}P(x_{k}|y=0)" title="P(x_{1},x_{2},...,x_{n}|y=0)=P(x_{1}|y=0)\cdot P(x_{2}|y=0)\cdot ...\cdot P(x_{n}|y=0)\\ =\prod_{k=1}^{n}P(x_{k}|y=0)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(x_{1},x_{2},...,x_{n}|y=1)=P(x_{1}|y=1)\cdot&space;P(x_{2}|y=1)\cdot&space;...\cdot&space;P(x_{n}|y=1)\\&space;=\prod_{k=1}^{n}P(x_{k}|y=1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x_{1},x_{2},...,x_{n}|y=1)=P(x_{1}|y=1)\cdot&space;P(x_{2}|y=1)\cdot&space;...\cdot&space;P(x_{n}|y=1)\\&space;=\prod_{k=1}^{n}P(x_{k}|y=1)" title="P(x_{1},x_{2},...,x_{n}|y=1)=P(x_{1}|y=1)\cdot P(x_{2}|y=1)\cdot ...\cdot P(x_{n}|y=1)\\ =\prod_{k=1}^{n}P(x_{k}|y=1)" /></a>

所以只需要比较$P(y=0)*\prod_{k=1}^{n}P(x_{k}|y=0)$和$P(y=1)*\prod_{k=1}^{n}P(x_{k}|y=1)$的大小就可以判断是否为垃圾邮件。
