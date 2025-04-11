# 1.代码部署

<img src="https://github.com/m00476/lxt/blob/master/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-01%20093512.png" width="800" >

# 2.优化特征选择方法

<img src="https://github.com/m00476/lxt/blob/master/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-01%20093622.png" width="800" >

---

# 代码核心功能说明

## 算法基础：多项式朴素贝叶斯分类器
- 条件概率独立性假设
多项式朴素贝叶斯假设特征（如词项）在给定类别条件下是相互独立的。即：

<img src="https://github.com/cat1-a/cat/blob/master/%E5%85%AC%E5%BC%8F1.png" width="200" >

其中 wi​ 表示词项，C表示类别。尽管现实中词项间存在关联，但这一简化假设显著降低了计算复杂度。

- 贝叶斯定理的应用形式
对于邮件分类任务，计算后验概率 P(C∣邮件内容)，选择最大概率的类别：

<img src="https://github.com/cat1-a/cat/blob/master/%E5%85%AC%E5%BC%8F2.png" width="200" >

P(C)：类别的先验概率（训练集中类别占比）。
P(w∣C)：词项 ww 在类别 C 中的条件概率（通过词频统计 + 拉普拉斯平滑计算）。

## 数据处理流程
- 分词处理：使用分词工具（如 jieba 中文分词或 nltk.word_tokenize 英文分词）将邮件文本拆分为词项列表。
- 停用词过滤：加载停用词表（如 nltk.corpus.stopwords 或自定义列表），过滤无意义词项。
- 预处理链：文件读取，无效字符过滤，中文分词，停用词过滤，文本标准化。

## 特征构建过程
- 方法对比

<img src="https://github.com/cat1-a/cat/blob/master/%E5%85%AC%E5%BC%8F3.png" width="500" >

- 核心差异
高频词：侧重区分高频词与低频词，可能受常见词（如“的”“是”）干扰。
TF-IDF：通过 IDF 降低常见词的权重，更关注类别区分性强的词（如专业术语）。

## 高频词/TF-IDF两种特征模式的切换方法
- 在代码中通过 参数配置 或 条件分支 切换特征提取器，示例如下：
$$(from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer)$$

# 配置参数（例如：feature_mode = 'high_freq' 或 'tfidf'）
$$(feature_mode = 't fidf')$$

$$(if feature_mode == 'high_freq':)$$
    $$(vectorizer = CountVectorizer(max_features=1000))$$  # 选择 Top 1000 高频词
$(elif feature_mode == 'tfidf':)$
    $$(vectorizer = TfidfVectorizer(max_features=1000))$$  # 使用 TF-IDF 加权

# 统一接口训练与预测
$$(X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts))$$

- 关键点
保持 vectorizer 的接口一致性（fit_transform/transform）。
调整 max_features 参数控制特征维度。

# 样本平衡处理

<img src="https://github.com/m00476/lxt/blob/master/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-01%20093634.png" width="800" >

# 增加模型评估指标

<img src="https://github.com/m00476/lxt/blob/master/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-01%20094015.png" width="800" >