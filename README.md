# machine-comprehension-ai-challenger-2018
## 简介
本数据集针对阅读理解中较为复杂的，需要利用整篇文章中多个句子的信息进行综合才能得到正确答案的观点型问题，构造了30万组由问题、篇章、候选答案组成的训练和测试集合。是目前为止全球难度最大的中文阅读理解公开数据集，全球最大的观点型机器阅读理解公开数据集。训练集：25万
验证集：3万
测试集A：1万
测试集B：1万
## 数据说明
每条数据为<问题，篇章，候选答案> 三元组组成。每个问题对应一个篇章（500字以内），以及包含正确答案的三个候选答案。
* 问题：真实用户自然语言问题，从搜索日志中随机选取并由机器初判后人工筛选
* 篇章：与问题对应的文本段，从问题相关的网页中人工选取
* 候选答案：人工生成的答案，提供若干（三个）选项，并标注正确答案
## 数据预览
数据以JSON格式表示如下样例：

{
            “query_id”:1,
            “query”:“维生c可以长期吃吗”,
            “url”: “https://wenwen.sogou.com/z/q748559425.htm”,
            “passage”: “每天吃的维生素的量没有超过推荐量的话是没有太大问题的。”,
            “alternatives”:”可以|不可以|无法确定”,
            “answer”:“可以”
        }
        
训练集给出上述全部字段，测试集不给answer字段
# 模型与实现
## 总体思路
本次比赛的题型和sQuAD不同。sQuAD是给出原文和问题，然后答案在原文中，模型需要给出答案在原文中的开始和结束位置。本次比赛是给出原文和问题，给出三个候选答案，模型需要给出一个最佳选项。模型总体借鉴BIDAF，修改output layer，让其输出一个词向量，然后用输出的词向量与候选答案中的三个词向量做相似度计算，然后将三个相似度归一化，看成三个概率，用cross entropy 做损失函数训练模型。这样做在验证集上训练第二轮后模型准确率达到67以上，但是后面就开始过拟合。后来我将BIDAF中contextual embed layer和modeling layer中的lstm全部换成了google在attention is all you need里的transformer的结构，但是验证集准确率依然没有上升。后来我发现我没有用position embedding，如果用了pe效果应该会好，但是最近都没什么时间再做。下面是BIDAF的结构介绍和几个我觉得重要地方的实现。参考2017年的在ICLR会议上发表的论文《BI-DIRECTIONAL ATTENSION FLOW FOR MACHINE COMPREHENSION》
## 关键结构和实现
该模型的结构图如下：
![image](https://github.com/zhongerqiandan/machine-comprehension-ai-challenger-2018/blob/master/1.png)
