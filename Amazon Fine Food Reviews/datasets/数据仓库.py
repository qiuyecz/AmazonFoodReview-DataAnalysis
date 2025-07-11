import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 添加这行
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 多备选方案
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.cluster import DBSCAN


# 初始化NLTK
nltk.download('vader_lexicon')

# ========== 数据加载 ==========
def load_data():
    """加载Amazon食品评论数据集"""
    df = pd.read_csv('.\\snap\\amazon-fine-food-reviews\\versions\\2\\Reviews.csv')
    df=df.head(20000)
    return df


# ========== 数据仓库构建 ==========
def build_data_warehouse(df):
    """构建数据仓库维度模型

    返回:
        df_fact: 事实表(包含评论ID、产品ID、用户ID、时间、评分等)
        df_product: 产品维度表
        df_user: 用户维度表
        df_date: 时间维度表
    """
    # 核心事实表
    df_fact = df[['Id', 'ProductId', 'UserId', 'Time', 'Score',
                  'HelpfulnessNumerator', 'HelpfulnessDenominator']]
    df_fact = df_fact.rename(columns={
        'Id': 'review_id',
        'Time': 'review_time'
    })
    df_fact['review_date'] = pd.to_datetime(df_fact['review_time'], unit='s')
    df_fact['date_id'] = df_fact['review_date'].dt.date

    # 商品维度
    df_product = df[['ProductId']].drop_duplicates()
    df_product['product_category'] = 'Food'

    # 用户维度
    df_user = df[['UserId', 'ProfileName']].drop_duplicates()

    # 时间维度
    date_range = pd.date_range(start=df_fact['review_date'].min(),
                               end=df_fact['review_date'].max())
    df_date = pd.DataFrame(date_range, columns=['date'])
    df_date['date_id'] = df_date['date'].dt.date
    df_date['year'] = df_date['date'].dt.year

    return df_fact, df_product, df_user, df_date



def clean_data(df):
    df = df.copy()

    # ===  缺失值处理 ===
    # [文本字段]
    # 文本字段填充
    df['Text'] = df['Text'].fillna('[MISSING]')
    df['Summary'] = df['Summary'].fillna('[MISSING]')

    # [关键ID字段]
    for col in ['Id', 'ProductId', 'UserId']:
        if df[col].isnull().sum() > 0:
            df = df.dropna(subset=[col])

    # [数值字段]
    # 有用性指标处理
    df['HelpfulnessDenominator'] = df['HelpfulnessDenominator'].fillna(0)
    df['HelpfulnessNumerator'] = df['HelpfulnessNumerator'].fillna(0)
    df['Score'] = df['Score'].fillna(df['Score'].median()).clip(1, 5)

    # === 异常值处理 ===
    # [评分异常]
    df['Score'] = np.where(
        (df['Score'] < 1) | (df['Score'] > 5),
        df.groupby('ProductId')['Score'].transform('median'),
        df['Score']
    )

    # [有用性比率]
    df['Helpfulness_Ratio'] = np.where(
        df['HelpfulnessDenominator'] <= 0,
        0,
        np.minimum(1, df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'])
    )

    # [时间异常]
    df['Review_Date'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
    df['Review_Date'] = df['Review_Date'].fillna(
        df.groupby('UserId')['Review_Date'].transform('median')
    )

    # 衍生特征
    df['Text_Length'] = df['Text'].str.len()
    df['Is_Short'] = df['Text_Length'] < 20
    df['Review_Day'] = df['Review_Date'].dt.date

    return df


def create_features(df):
    """创建衍生特征"""
    df = df.copy()
    # 文本长度特征
    df['text_length'] = df['Text'].apply(len)
    df['summary_length'] = df['Summary'].apply(len)

    # 情感分析特征
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['Text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df


# ========== 用户分层分析 ==========
def rfm_analysis_Kmeans(df_fact, df_with_sentiment):
    """基于RFM模型的用户分层分析

    参数:
        df_fact: 事实表数据
        df_with_sentiment: 包含情感分析结果的原始数据

    返回:
        df_rfm: 包含用户分群结果的DataFrame
    """
    # 合并情感分析结果
    df_merged = df_fact.merge(
        df_with_sentiment[['Id', 'sentiment']],
        left_on='review_id',
        right_on='Id',
        how='left'
    ).drop('Id', axis=1)

    # 计算RFM指标
    df_rfm = df_merged.groupby('UserId').agg({
        'review_date': lambda x: (df_merged['review_date'].max() - x.max()).days,
        'review_id': 'count',
        'Score': 'mean',
        'sentiment': 'mean'
    }).rename(columns={
        'review_date': 'recency',
        'review_id': 'frequency',
        'Score': 'avg_score'
    }).dropna()  # 去除NaN值

    # 标准化
    scaler = StandardScaler()
    df_rfm_scaled = scaler.fit_transform(df_rfm)

    # 寻找最佳K值
    range_k = range(2, 8)
    scores = []
    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df_rfm_scaled)
        scores.append(silhouette_score(df_rfm_scaled, labels))

    # 可视化肘部法则
    plt.figure(figsize=(10, 4))
    plt.plot(range_k, scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal K Value Selection')
    plt.show()

    # 选择最佳K值
    best_k = range_k[np.argmax(scores)]
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df_rfm['cluster'] = kmeans.fit_predict(df_rfm_scaled)

    # 分群特征分析
    cluster_profile = df_rfm.groupby('cluster').agg({
        'recency': ['mean', 'median'],
        'frequency': ['mean', 'median'],
        'avg_score': 'mean',
        'sentiment': 'mean'
    })

    print("=== 用户分群特征 ===")
    print(cluster_profile)

    # 可视化
    sns.pairplot(df_rfm, hue='cluster', vars=['recency', 'frequency', 'avg_score'])
    plt.suptitle('User Clusters Visualization', y=1.02)
    plt.show()

    return df_rfm


def rfm_analysis_dbscan(df_fact, df_with_sentiment):
    """基于RFM模型的用户分层分析（使用DBSCAN）"""
    # 合并情感分析结果
    df_merged = df_fact.merge(
        df_with_sentiment[['Id', 'sentiment']],
        left_on='review_id',
        right_on='Id',
        how='left'
    ).drop('Id', axis=1)

    # 计算RFM指标
    df_rfm = df_merged.groupby('UserId').agg({
        'review_date': lambda x: (df_merged['review_date'].max() - x.max()).days,
        'review_id': 'count',
        'Score': 'mean',
        'sentiment': 'mean'
    }).rename(columns={
        'review_date': 'recency',
        'review_id': 'frequency',
        'Score': 'avg_score'
    }).dropna()

    # 标准化
    scaler = StandardScaler()
    df_rfm_scaled = scaler.fit_transform(df_rfm)

    # 寻找最佳DBSCAN参数（eps和min_samples）
    # 通过K距离图选择eps
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(df_rfm_scaled)
    distances, indices = nbrs.kneighbors(df_rfm_scaled)
    k_distances = np.sort(distances[:, -1], axis=0)

    # 绘制K距离图
    plt.figure(figsize=(10, 4))
    plt.plot(k_distances)
    plt.xlabel('Points')
    plt.ylabel('5th Nearest Neighbor Distance')
    plt.title('K-Distance Graph for DBSCAN')
    plt.show()

    # 根据K距离图选择eps
    eps = 1.5  # 需要根据实际K距离图调整
    min_samples = 5  # 最小样本数

    # 使用DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_rfm['cluster'] = dbscan.fit_predict(df_rfm_scaled)

    # 过滤噪声点（cluster=-1）
    df_rfm_clean = df_rfm[df_rfm['cluster'] != -1]

    # 计算轮廓系数
    if len(np.unique(df_rfm_clean['cluster'])) > 1:
        silhouette_avg = silhouette_score(df_rfm_scaled[df_rfm['cluster'] != -1], df_rfm_clean['cluster'])
        print(f"Silhouette Score: {silhouette_avg:.2f}")
    else:
        print("Only one cluster found (excluding noise).")

    # 分群特征分析
    cluster_profile = df_rfm_clean.groupby('cluster').agg({
        'recency': ['mean', 'median'],
        'frequency': ['mean', 'median'],
        'avg_score': 'mean',
        'sentiment': 'mean'
    })

    print("=== 用户分群特征（排除噪声点） ===")
    print(cluster_profile)

    # 可视化
    sns.pairplot(df_rfm_clean, hue='cluster', vars=['recency', 'frequency', 'avg_score'])
    plt.suptitle('User Clusters Visualization (DBSCAN)', y=1.02)
    plt.show()

    return df_rfm


# ========== 运营建议输出 ==========
def generate_recommendations(df_rfm, method='kmeans'):
    """根据用户分群生成运营建议（"""
    # 计算各群用户占比
    cluster_stats = df_rfm['cluster'].value_counts(normalize=True).reset_index()
    cluster_stats.columns = ['cluster', 'percentage']

    # 根据聚类方法定义分群策略
    if method == 'kmeans':
        # K-means分群定义
        recommendations = [
            {
                'cluster': 0,
                '分群名称': '高价值活跃用户',
                '特征描述': '近期活跃+高频+高评分+积极情感',
                '运营策略': ('专属VIP通道\n新品优先试用\n高价值用户奖励计划')
            },
            {
                'cluster': 1,
                '分群名称': '不满意用户',
                '特征描述': '较近期活跃+低频+低评分+中性情感',
                '运营策略': ('满意度回访\n问题产品退换\n定向优惠挽回')
            },
            {
                'cluster': 2,
                '分群名称': '潜力用户',
                '特征描述': '最活跃+最高频+高评分+积极情感',
                '运营策略': ('忠诚度计划\n交叉销售推荐\n限时升级奖励')
            }
        ]
    else:  # DBSCAN分群定义
        recommendations = [
            {
                'cluster': -1,
                '分群名称': '噪声点/特殊用户',
                '特征描述': '不符合主流模式（数据异常或样本量极少）',
                '运营策略': ('单独行为分析\n异常交易监控\n人工审核')
            },
            {
                'cluster': 0,
                '分群名称': '低频忠诚用户',
                '特征描述': '活跃度中等+低频购买+高评分+积极情感',
                '运营策略': ('定向新品推送\n会员等级提升激励\n专属折扣券发放')
            },
            {
                'cluster': 1,
                '分群名称': '高频超级用户',
                '特征描述': '高活跃度+高频购买+超高评分+极积极情感',
                '运营策略': ('邀请加入产品共创\n专属客服通道\n年度消费返利')
            },
            {
                'cluster': 2,
                '分群名称': '中频活跃用户',
                '特征描述': '活跃度中等+中频购买+高评分+积极情感',
                '运营策略': ('消费满减活动\n积分加速计划\n相关产品推荐')
            },
            {
                'cluster': 3,
                '分群名称': '高频活跃用户',
                '特征描述': '高活跃度+高频购买+高评分+积极情感',
                '运营策略': ('会员权益升级\n限时特惠优先\n专属活动邀请')
            },
            {
                'cluster': 4,
                '分群名称': '中频忠诚用户',
                '特征描述': '活跃度中等+中频购买+高评分+积极情感',
                '运营策略': ('周期性促销提醒\n积分兑换优惠\n品牌活动邀请')
            },
            {
                'cluster': 5,
                '分群名称': '超高频超级用户',
                '特征描述': '极高活跃度+超高频购买+超高评分+极积极情感',
                '运营策略': ('私人定制服务\n品牌大使计划\n终身VIP权益')
            },
            {
                'cluster': 6,
                '分群名称': '低频高价值用户',
                '特征描述': '活跃度较低+低频购买+高评分+积极情感',
                '运营策略': ('唤醒营销活动\n专属福利召回\n个性化推荐')
            }
        ]

    # 筛选实际存在的cluster并排序
    existing_clusters = df_rfm['cluster'].unique()
    filtered_rec = [rec for rec in recommendations if rec['cluster'] in existing_clusters]
    filtered_rec.sort(key=lambda x: x['cluster'])  # 按clusterID排序

    # 创建运营建议DataFrame
    strategy_table = pd.DataFrame(filtered_rec)

    # 合并占比数据
    strategy_table = strategy_table.merge(
        cluster_stats,
        on='cluster',
        how='left'
    ).fillna({
        '分群名称': '未知群组',
        '特征描述': '数据不足需补充分析',
        '运营策略': '暂不采取行动'
    })

    # 打印运营建议表格
    print("\n=== 用户分群运营策略建议 ===")
    print("=" * 90)
    print(f"{'群组ID':<8}{'分群名称':<18}{'占比':<8}{'特征描述':<45}{'运营策略'}")
    print("-" * 90)
    for _, row in strategy_table.iterrows():
        print(
            f"{row['cluster']:<8}{row['分群名称']:<18}{row['percentage']:.1%}{' ':>2}{row['特征描述']:<45}{row['运营策略']}")
    print("=" * 90)

    # 可视化用户分布
    plt.figure(figsize=(14, 7))
    if method == 'kmeans':
        palette = ['#2ecc71', '#e74c3c', '#3498db']  # 绿色,红色,蓝色
    else:

        palette = sns.color_palette("Set3", n_colors=len(strategy_table))  # 使用Set3色板

    bar_plot = sns.barplot(
        x='分群名称',
        y='percentage',
        hue='分群名称',
        data=strategy_table,
        palette=palette,
        legend=False
    )


    for p in bar_plot.patches:
        height = p.get_height()
        bar_plot.text(p.get_x() + p.get_width() / 2., height + 0.005,
                      f'{height:.1%}', ha='center', fontsize=9)

    plt.title(f'用户分群占比分布 ({method.upper()}方法)', fontsize=14)
    plt.ylabel('占比', fontsize=12)
    plt.xticks(rotation=30, ha='right')  # 优化x轴标签显示
    plt.tight_layout()
    plt.show()

# ========== 主流程 ==========
def main():
    # 数据加载
    df = load_data()

    # 数据清洗
    df_clean = clean_data(df)

    # 特征工程
    df_features = create_features(df_clean)

    # 构建数据仓库
    df_fact, _, _, _ = build_data_warehouse(df_clean)

    # K-means用户分层分析
    df_rfm_kmeans = rfm_analysis_Kmeans(df_fact, df_features)

    # K-means结果
    generate_recommendations(df_rfm_kmeans, method='kmeans')

    # DBSCAN用户分层分析
    df_rfm_dbscan = rfm_analysis_dbscan(df_fact, df_features)

    # DBSCAN结果
    generate_recommendations(df_rfm_dbscan, method='dbscan')



if __name__ == '__main__':
    df_rfm_result = main()
