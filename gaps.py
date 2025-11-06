import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Streamlit页面配置
st.set_page_config(page_title="Gaps Indicator", page_icon="📊", layout="wide")
st.title("📈 Gaps Indicator - 价格缺口检测与可视化")

# 侧边栏参数设置
st.sidebar.header("参数设置")
ticker = st.sidebar.text_input("股票代码", value="TSLA", help="输入股票代码，例如: TSLA")
period = st.sidebar.selectbox("数据周期", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
gap_threshold = st.sidebar.slider("缺口阈值 (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, 
                                  help="最小缺口百分比，用于过滤小缺口")
show_alerts = st.sidebar.checkbox("启用警报", value=True)
show_partial_close = st.sidebar.checkbox("显示部分关闭", value=True)
show_full_close = st.sidebar.checkbox("显示完全关闭", value=True)

# 获取股票数据
@st.cache_data
def load_data(ticker, period):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            st.error("无法获取数据，请检查股票代码。")
            return None
        return data
    except Exception as e:
        st.error(f"数据加载错误: {e}")
        return None

data = load_data(ticker, period)
if data is not None:
    # 计算缺口
    data['Prev_Close'] = data['Close'].shift(1)
    data['Gap_Size'] = ((data['Open'] - data['Prev_Close']) / data['Prev_Close']) * 100
    data['Gap_Type'] = np.where(data['Gap_Size'] > gap_threshold, 'Up', 
                                np.where(data['Gap_Size'] < -gap_threshold, 'Down', 'None'))
    data['Has_Gap'] = data['Gap_Type'] != 'None'
    
    # 检测缺口关闭
    gaps = data[data['Has_Gap']].copy()
    if not gaps.empty:
        # 为每个缺口跟踪关闭状态
        data['Gap_Close_Status'] = 'Open'
        for idx, gap_row in gaps.iterrows():
            gap_start = gap_row.name
            gap_end = gap_row['Prev_Close'] if gap_row['Gap_Type'] == 'Up' else gap_row['Open']
            gap_start_price = gap_row['Open'] if gap_row['Gap_Type'] == 'Up' else gap_row['Prev_Close']
            
            # 检查后续价格是否进入缺口（部分关闭）
            post_gap_data = data.loc[gap_start:].copy()
            for future_idx, future_row in post_gap_data.iterrows():
                if future_idx > gap_start:
                    low = future_row['Low']
                    high = future_row['High']
                    
                    if show_partial_close:
                        # 部分关闭: 价格触及缺口区域
                        if gap_row['Gap_Type'] == 'Up':
                            if low <= gap_end:
                                data.loc[future_idx, 'Gap_Close_Status'] = 'Partial'
                                break
                        else:  # Down gap
                            if high >= gap_start:
                                data.loc[future_idx, 'Gap_Close_Status'] = 'Partial'
                                break
                    
                    # 完全关闭: 价格穿越缺口
                    if show_full_close:
                        if gap_row['Gap_Type'] == 'Up':
                            if future_row['Close'] <= gap_end:
                                data.loc[future_idx, 'Gap_Close_Status'] = 'Full'
                                break
                        else:  # Down gap
                            if future_row['Close'] >= gap_start:
                                data.loc[future_idx, 'Gap_Close_Status'] = 'Full'
                                break

    # 可视化
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=['价格缺口图表'],
                        row_width=[0.2])

    # 添加蜡烛图
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='OHLC'),
                  row=1, col=1)

    # 绘制缺口矩形
    active_gaps = data[data['Has_Gap'] & (data['Gap_Close_Status'] == 'Open')]
    partial_gaps = data[data['Has_Gap'] & (data['Gap_Close_Status'] == 'Partial')]
    full_gaps = data[data['Has_Gap'] & (data['Gap_Close_Status'] == 'Full')]

    def add_gap_rectangles(gap_data, color, opacity, label):
        for idx, row in gap_data.iterrows():
            if row['Gap_Type'] == 'Up':
                y0 = row['Prev_Close']
                y1 = row['Open']
            else:
                y0 = row['Open']
                y1 = row['Prev_Close']
            
            fig.add_shape(type="rect",
                          x0=idx - timedelta(days=0.5), x1=data.index[-1],
                          y0=min(y0, y1), y1=max(y0, y1),
                          fillcolor=color, opacity=opacity,
                          line=dict(color=color, width=1),
                          name=label,
                          row=1, col=1)

    # 绘制活跃缺口 (红色，半透明)
    add_gap_rectangles(active_gaps, 'rgba(255, 0, 0, 0.3)', 0.3, 'Active Gap')

    # 部分关闭缺口 (橙色，更透明)
    if show_partial_close:
        add_gap_rectangles(partial_gaps, 'rgba(255, 165, 0, 0.2)', 0.2, 'Partial Close')

    # 完全关闭缺口 (绿色，最透明)
    if show_full_close:
        add_gap_rectangles(full_gaps, 'rgba(0, 255, 0, 0.2)', 0.2, 'Full Close')

    # 更新布局
    fig.update_layout(yaxis_title='价格 (USD)', xaxis_title='日期',
                      title=f"{ticker} 价格缺口分析 ({period})",
                      height=600, showlegend=False,
                      hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # 缺口统计表格
    st.subheader("缺口统计")
    gap_stats = data[data['Has_Gap']].groupby('Gap_Type').agg({
        'Gap_Size': ['count', 'mean', 'min', 'max']
    }).round(2)
    gap_stats.columns = ['数量', '平均大小 (%)', '最小 (%)', '最大 (%)']
    st.table(gap_stats)

    # 警报
    if show_alerts:
        st.subheader("警报")
        recent_data = data.tail(5)
        for idx, row in recent_data.iterrows():
            if row['Has_Gap']:
                gap_dir = "向上" if row['Gap_Type'] == 'Up' else "向下"
                st.warning(f"🚨 新{ gap_dir }缺口检测! 大小: {abs(row['Gap_Size']):.2f}% (日期: {idx.date()})")
            elif 'Partial' in row.get('Gap_Close_Status', ''):
                st.info(f"ℹ️ 部分关闭缺口 (日期: {idx.date()})")
            elif 'Full' in row.get('Gap_Close_Status', ''):
                st.success(f"✅ 完全关闭缺口 (日期: {idx.date()})")

    # 数据下载
    csv = data.to_csv()
    st.download_button("下载数据 (CSV)", csv, f"{ticker}_gaps_{period}.csv", "text/csv")
