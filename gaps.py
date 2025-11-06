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
        # 修复 yfinance 最近版本返回 MultiIndex columns 的问题（针对单只股票）
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
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
    
    # 检测缺口关闭 - 改进逻辑：为每个缺口独立跟踪状态
    gaps = data[data['Has_Gap']].copy().reset_index()
    gap_status = {}  # {gap_date: status}
    close_dates = {}  # {gap_date: close_date}
    
    for _, gap_row in gaps.iterrows():
        gap_date = gap_row['index']  # 原索引（日期）
        gap_type = gap_row['Gap_Type']
        gap_end = gap_row['Prev_Close'] if gap_type == 'Up' else gap_row['Open']
        gap_start_price = gap_row['Open'] if gap_type == 'Up' else gap_row['Prev_Close']
        
        status = 'Open'
        close_date = None
        
        # 检查后续价格是否进入缺口
        post_gap_data = data.loc[gap_date:].iloc[1:]  # 从下一天开始
        for future_date, future_row in post_gap_data.iterrows():
            low = future_row['Low']
            high = future_row['High']
            close = future_row['Close']
            
            partial_closed = False
            full_closed = False
            
            if gap_type == 'Up':
                # Up gap: 填充从上方下降
                if show_partial_close and low <= gap_end:
                    partial_closed = True
                if show_full_close and close <= gap_end:
                    full_closed = True
            else:  # Down gap
                # Down gap: 填充从下方上升
                if show_partial_close and high >= gap_start_price:
                    partial_closed = True
                if show_full_close and close >= gap_start_price:
                    full_closed = True
            
            if partial_closed or full_closed:
                if full_closed:
                    status = 'Full'
                else:
                    status = 'Partial'
                close_date = future_date
                break  # 一旦关闭，就停止检查
        
        gap_status[gap_date] = status
        close_dates[gap_date] = close_date
    
    # 将状态合并回数据（仅用于过滤和警报）
    data['Gap_Close_Status'] = data.index.map(lambda x: gap_status.get(x, 'Open') if data.loc[x, 'Has_Gap'] else 'N/A')
    
    # 按状态分组缺口数据
    active_gaps = gaps[gaps['index'].map(gap_status) == 'Open']
    partial_gaps = gaps[gaps['index'].map(gap_status) == 'Partial']
    full_gaps = gaps[gaps['index'].map(gap_status) == 'Full']

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
    def add_gap_rectangles(gap_df, color, opacity, label):
        for _, row in gap_df.iterrows():
            gap_date = row['index']
            gap_type = row['Gap_Type']
            if gap_type == 'Up':
                y0 = row['Prev_Close']
                y1 = row['Open']
            else:
                y0 = row['Open']
                y1 = row['Prev_Close']
            
            # 矩形从缺口日延伸到结束或关闭日
            x1 = close_dates.get(gap_date, data.index[-1]) if gap_status.get(gap_date) != 'Open' else data.index[-1]
            fig.add_shape(type="rect",
                          x0=gap_date - timedelta(days=0.5), x1=x1,
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

    # 警报 - 基于最近缺口
    if show_alerts:
        st.subheader("警报")
        recent_gaps = gaps.tail(5)  # 最近5个缺口
        for _, row in recent_gaps.iterrows():
            gap_date = row['index']
            status = gap_status.get(gap_date, 'Open')
            gap_dir = "向上" if row['Gap_Type'] == 'Up' else "向下"
            gap_size = abs(row['Gap_Size'])
            
            if status == 'Open':
                st.warning(f"🚨 新{ gap_dir }缺口检测! 大小: {gap_size:.2f}% (日期: {gap_date.date()})")
            elif status == 'Partial':
                close_date = close_dates.get(gap_date)
                st.info(f"ℹ️ 部分关闭{ gap_dir }缺口 (大小: {gap_size:.2f}%, 关闭日期: {close_date.date() if close_date else 'N/A'})")
            elif status == 'Full':
                close_date = close_dates.get(gap_date)
                st.success(f"✅ 完全关闭{ gap_dir }缺口 (大小: {gap_size:.2f}%, 关闭日期: {close_date.date() if close_date else 'N/A'})")

    # 数据下载
    csv = data.to_csv()
    st.download_button("下载数据 (CSV)", csv, f"{ticker}_gaps_{period}.csv", "text/csv")
