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

# 新增：交易策略参数
st.sidebar.header("交易策略设置")
enable_strategy = st.sidebar.checkbox("启用缺口交易策略", value=True)
strategy_type = st.sidebar.selectbox("策略类型", ["简单缺口填补", "缺口延续"], index=0)
position_size = st.sidebar.slider("仓位大小 (%)", min_value=1.0, max_value=100.0, value=100.0, step=10.0,
                                  help="每次交易的仓位百分比（初始资金100%）")
stop_loss_pct = st.sidebar.slider("止损 (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.5,
                                  help="基于缺口大小的止损百分比")

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
    gaps = data[data['Has_Gap']].copy()
    gap_status = {}  # {gap_date: status}
    close_dates = {}  # {gap_date: close_date}
    
    for idx, gap_row in gaps.iterrows():
        gap_date = idx
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
            
            # 始终检测部分和完全关闭条件（不受显示选项影响）
            if gap_type == 'Up':
                partial_cond = low <= gap_end
                full_cond = close <= gap_end
            else:  # Down gap
                partial_cond = high >= gap_start_price
                full_cond = close >= gap_start_price
            
            if full_cond:
                status = 'Full'
                close_date = future_date
                break
            elif partial_cond:
                status = 'Partial'
                close_date = future_date
                break
        
        gap_status[gap_date] = status
        close_dates[gap_date] = close_date
    
    # 将状态合并回数据
    data['Gap_Close_Status'] = 'N/A'
    for date, stat in gap_status.items():
        data.loc[date, 'Gap_Close_Status'] = stat
    
    # 按状态分组缺口数据
    active_gaps = data[data['Has_Gap'] & (data['Gap_Close_Status'] == 'Open')]
    partial_gaps = data[data['Has_Gap'] & (data['Gap_Close_Status'] == 'Partial')]
    full_gaps = data[data['Has_Gap'] & (data['Gap_Close_Status'] == 'Full')]

    # 新增：缺口交易策略回测
    if enable_strategy:
        # 初始化策略列
        data['Strategy_Signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
        data['Position'] = 0  # 当前仓位: 1: 多头, -1: 空头, 0: 无仓
        data['Entry_Price'] = np.nan
        data['Exit_Price'] = np.nan
        data['Strategy_Return'] = 0.0
        data['Cumulative_Return'] = 0.0
        data['Trades'] = []  # 记录交易
        
        initial_capital = 10000  # 初始资金
        capital = initial_capital
        position = 0
        entry_price = 0
        
        for i in range(1, len(data)):
            current_date = data.index[i]
            prev_date = data.index[i-1]
            row = data.iloc[i]
            
            # 生成信号
            signal = 0
            if row['Has_Gap']:
                if strategy_type == "简单缺口填补":
                    # 填补策略: Up Gap 做空（期待填补），Down Gap 做多
                    if row['Gap_Type'] == 'Up':
                        signal = -1  # 卖出（空头）
                    elif row['Gap_Type'] == 'Down':
                        signal = 1   # 买入（多头）
                elif strategy_type == "缺口延续":
                    # 延续策略: Up Gap 做多，Down Gap 做空
                    if row['Gap_Type'] == 'Up':
                        signal = 1   # 买入
                    elif row['Gap_Type'] == 'Down':
                        signal = -1  # 卖出
            
            data.iloc[i, data.columns.get_loc('Strategy_Signal')] = signal
            
            # 仓位管理
            if signal != 0 and position == 0:
                # 开仓
                position = signal
                entry_price = row['Open']
                data.iloc[i, data.columns.get_loc('Entry_Price')] = entry_price
                data.iloc[i, data.columns.get_loc('Position')] = position
                trades = data.iloc[i, data.columns.get_loc('Trades')]
                trades.append({'date': current_date, 'action': 'entry', 'price': entry_price, 'type': row['Gap_Type']})
            
            elif position != 0:
                # 检查平仓条件: 缺口关闭 或 止损
                close_status = gap_status.get(prev_date, 'Open') if prev_date in gap_status else 'Open'
                exit_signal = False
                exit_price = row['Open']
                
                if close_status in ['Partial', 'Full']:
                    exit_signal = True
                elif stop_loss_pct > 0:
                    pnl_pct = ((row['Open'] - entry_price) / entry_price) * position
                    if pnl_pct <= -stop_loss_pct / 100:
                        exit_signal = True
                
                if exit_signal:
                    # 平仓
                    exit_price = row['Open']
                    data.iloc[i, data.columns.get_loc('Exit_Price')] = exit_price
                    data.iloc[i, data.columns.get_loc('Position')] = 0
                    
                    # 计算回报
                    trade_return = ((exit_price - entry_price) / entry_price) * position * (position_size / 100)
                    data.iloc[i, data.columns.get_loc('Strategy_Return')] = trade_return
                    capital *= (1 + trade_return)
                    
                    trades = data.iloc[i, data.columns.get_loc('Trades')]
                    trades.append({'date': current_date, 'action': 'exit', 'price': exit_price, 'pnl': trade_return})
                    position = 0
                    entry_price = 0
            
            else:
                data.iloc[i, data.columns.get_loc('Position')] = position
        
        # 累计回报
        data['Cumulative_Return'] = (capital / initial_capital - 1) * 100
        final_return = data['Cumulative_Return'].iloc[-1]
        
        # 策略绩效统计
        trades_df = pd.DataFrame(data['Trades'].iloc[-1]) if data['Trades'].iloc[-1] else pd.DataFrame()
        num_trades = len(trades_df) // 2 if not trades_df.empty else 0
        win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df[trades_df['action'] == 'exit']) if len(trades_df[trades_df['action'] == 'exit']) > 0 else 0

    # 可视化 - 主图: 价格缺口 + 策略信号
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=['价格缺口图表', '策略权益曲线'],
                        row_width=[0.2, 0.7])

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
        for idx, row in gap_df.iterrows():
            gap_date = idx
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

    # 添加策略信号标记
    if enable_strategy:
        buy_signals = data[data['Strategy_Signal'] == 1]
        sell_signals = data[data['Strategy_Signal'] == -1]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'] * 0.98,
                                 mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                                 name='买入信号'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'] * 1.02,
                                 mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                                 name='卖出信号'), row=1, col=1)

    # 添加缺口定义注解
    annotations = []
    
    # Up Gap 定义
    annotations.append(dict(
        xref='paper', yref='paper',
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        text='上缺口（Up Gap）: 当前 K 线的开盘价（或低点）高于前一根 K 线的收盘价（或高点），表示强势上涨（牛市信号）。',
        showarrow=False,
        font=dict(size=10, color='green'),
        bgcolor='rgba(0,255,0,0.1)',
        bordercolor='green',
        borderwidth=1,
        row=1, col=1
    ))
    
    # Down Gap 定义
    annotations.append(dict(
        xref='paper', yref='paper',
        x=0.02, y=0.92,
        xanchor='left', yanchor='top',
        text='下缺口（Down Gap）: 当前 K 线的开盘价（或高点）低于前一根 K 线的收盘价（或低点），表示强势下跌（熊市信号）。',
        showarrow=False,
        font=dict(size=10, color='red'),
        bgcolor='rgba(255,0,0,0.1)',
        bordercolor='red',
        borderwidth=1,
        row=1, col=1
    ))

    # 更新主图布局
    fig.update_layout(yaxis_title='价格 (USD)', xaxis_title='日期',
                      title=f"{ticker} 价格缺口分析 ({period})",
                      height=800, showlegend=True,
                      hovermode='x unified',
                      annotations=annotations)
    fig.update_xaxes(rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # 策略绩效图（如果启用）
    if enable_strategy:
        # 权益曲线
        equity_curve = pd.Series(index=data.index, data=np.cumsum(data['Strategy_Return']) * initial_capital + initial_capital)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=equity_curve, mode='lines', name='策略权益'))
        fig2.add_trace(go.Scatter(x=data.index, y=data['Close'] / data['Close'].iloc[0] * initial_capital, 
                                  mode='lines', name='买入并持有'))
        fig2.update_layout(title=f"{strategy_type} 策略权益曲线 (最终回报: {final_return:.2f}%)",
                           yaxis_title='权益 (USD)', xaxis_title='日期', height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # 策略统计
        st.subheader("策略绩效统计")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总交易次数", num_trades)
        col2.metric("胜率 (%)", f"{win_rate * 100:.1f}")
        col3.metric("总回报 (%)", f"{final_return:.2f}")
        col4.metric("最大回撤 (%)", "N/A")  # 可进一步计算

        # 交易列表
        if not trades_df.empty:
            st.subheader("交易记录")
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            st.dataframe(trades_df)

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
        recent_gaps = data[data['Has_Gap']].tail(5)
        for idx, row in recent_gaps.iterrows():
            gap_date = idx
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
