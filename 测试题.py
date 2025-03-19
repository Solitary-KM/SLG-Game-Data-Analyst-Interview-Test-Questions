import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter

# 初始化NLTK资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


# 数据加载与预处理
def load_data(file_path):
    # 处理可能的编码问题和列名
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='gbk')

    # 重命名列
    df = df.rename(columns={
        "content": "message",
        "事件时间": "timestamp",
        "role_id": "user_id"
    })

    # 时间格式转换
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])  # 删除无效时间记录

    return df


# 文本清洗与分词
def clean_text(text):
    if pd.isna(text): return []
    text = re.sub(r'[^\w\s]', '', str(text).lower())  # 去除非字母字符
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words and len(word) > 2]
    return filtered


# 高频话题分析
def analyze_topics(df):
    all_words = [word for tokens in df['cleaned_text'] for word in tokens]
    word_freq = Counter(all_words)

    # 生成词云
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top Topics in Chat Messages')
    plt.savefig('wordcloud.png', bbox_inches='tight')
    plt.close()

    return pd.DataFrame(word_freq.most_common(20), columns=['Keyword', 'Frequency'])


# 用户活跃时段分析
def analyze_activity(df):
    df['hour'] = df['timestamp'].dt.hour
    hourly_activity = df.groupby('hour').size()

    # 绘制时段分布
    plt.figure(figsize=(12, 6))
    hourly_activity.plot(kind='bar', color='#1f77b4')
    plt.title('Hourly Activity Distribution (UTC)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Message Count')
    plt.savefig('hourly_activity.png', dpi=300)
    plt.close()

    return hourly_activity


# 情感分析与流失原因挖掘
def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['message'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    # 提取负面消息中的关键词
    negative_msgs = df[df['sentiment'] < -0.5]
    negative_words = [word for tokens in negative_msgs['cleaned_text'] for word in tokens]

    return pd.DataFrame(Counter(negative_words).most_common(10), columns=['Negative_Word', 'Count'])



if __name__ == "__main__":
    # 加载数据
    df = load_data('聊天.csv')

    # 文本清洗
    df['cleaned_text'] = df['message'].apply(clean_text)

    # 分析流程
    topic_df = analyze_topics(df)
    activity_series = analyze_activity(df)
    sentiment_df = analyze_sentiment(df)

    # 保存结果
    topic_df.to_csv('top_topics.csv', index=False)
    activity_series.to_csv('hourly_activity.csv', header=['Message_Count'])
    sentiment_df.to_csv('negative_keywords.csv', index=False)

    # 打印关键结果
    print("【主要关注话题】\n", topic_df)
    print("\n【用户活跃高峰时段】\n", activity_series.idxmax(), f"点（消息量：{activity_series.max()}条）")
    print("\n【负面关键词Top10】\n", sentiment_df)