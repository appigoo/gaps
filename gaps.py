import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from ta.trend import ADXIndicator  # 新增: ta库用于ADX

# Streamlit页面配置
st.set_page_config(page_title="Gaps Indicator", page_icon="📊", layout="wide")
st.title("📈 Gaps Indicator - 价格缺口检测与可视化（集成ML预测）")

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

# 改进：信号准确度参数
st.sidebar.header("信号准确度过滤")
volume_multiplier = st.sidebar.slider("成交量过滤倍数", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                                      help="成交量需超过平均值的倍数才触发信号")
ml_threshold = st.sidebar.slider("ML预测阈值 (%)", min_value=50.0, max_value=90.0, value=70.0, step=5.0,
                                 help="ML预测概率超过阈值才确认信号（仅当启用ML时）")
adx_threshold = st.sidebar.slider("ADX趋势强度阈值", min_value=20.0, max_value=40.0, value=25.0, step=1.0,
                                  help="ADX > 阈值表示强趋势，增强延续策略信号")

# 新增：ML预测参数
st.sidebar.header("ML预测设置")
enable_ml = st.sidebar.checkbox("启用ML缺口预测", value=True)
ml_model_type = st.sidebar.selectbox("ML模型类型", ["LSTM (时间序列)", "MLP (多层感知器)"], index=0)
prediction_horizon = st.sidebar.slider("预测天数", min_value=1, max_value=10, value=5, help="预测未来缺口概率")

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
    
    # 特征工程：为ML准备
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(5).std()
    data['MA_5'] = data['Close'].rolling(5).mean()
    data['MA_20'] = data['Close'].rolling(20).mean()
    data['RSI'] = compute_rsi(data['Close'], 14)  # 自定义RSI函数
    data['Volume_MA'] = data['Volume'].rolling(20).mean()  # 新增: 平均成交量
    data['ADX'] = ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()  # 新增: ADX趋势强度
    data['Target'] = np.where(data['Gap_Type'].shift(-1) == 'Up', 1, 
                              np.where(data['Gap_Type'].shift(-1) == 'Down', -1, 0))  # 下一天缺口标签: 1=Up, -1=Down, 0=None
    
    # 填充NaN
    data = data.fillna(method='ffill').fillna(0)
    
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

    # 新增：ML预测模型训练与预测
    ml_predictions = None
    ml_model = None
    scaler = None
    if enable_ml:
        # 准备特征（新增ADX和Volume相关）
        features = ['Returns', 'Volatility', 'MA_5', 'MA_20', 'RSI', 'Gap_Size', 'Volume_MA', 'ADX']
        X = data[features].dropna()
        y = data['Target'].loc[X.index]  # 对应标签
        
        if len(X) > 20:  # 确保足够数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 转换为Tensor
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
            
            # 数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 定义模型
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size=50, num_layers=1, num_classes=3):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, num_classes)
                
                def forward(self, x):
                    # x shape: (batch, seq_len=1, features)
                    h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
                    c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
                    out, _ = self.lstm(x.unsqueeze(1), (h0, c0))  # 扩展seq_len=1
                    out = self.fc(out[:, -1, :])
                    return out
            
            class MLPModel(nn.Module):
                def __init__(self, input_size, hidden_size=50, num_classes=3):
                    super(MLPModel, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.fc2 = nn.Linear(hidden_size, num_classes)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    out = self.relu(self.fc1(x))
                    out = self.fc2(out)
                    return out
            
            # 选择模型
            if ml_model_type == "LSTM (时间序列)":
                ml_model = LSTMModel(input_size=X.shape[1])
            else:
                ml_model = MLPModel(input_size=X.shape[1])
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(ml_model.parameters(), lr=0.001)
            
            # 训练
            ml_model.train()
            for epoch in range(50):  # 简单训练50 epochs
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = ml_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # 预测
            ml_model.eval()
            with torch.no_grad():
                test_outputs = ml_model(X_test_tensor)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = (predicted == y_test_tensor).float().mean().item()
            
            st.info(f"ML模型训练完成。测试准确率: {accuracy:.2%}")
            
            # 未来预测：使用最近数据预测未来prediction_horizon天
            recent_features = data[features].tail(prediction_horizon * 2).dropna()  # 最近数据
            if len(recent_features) > 0:
                recent_scaled = scaler.transform(recent_features)
                recent_tensor = torch.tensor(recent_scaled, dtype=torch.float32)
                with torch.no_grad():
                    pred_outputs = ml_model(recent_tensor)
                    pred_probs = torch.softmax(pred_outputs, dim=1).numpy()
                    ml_predictions = pd.DataFrame(pred_probs, columns=['None', 'Down', 'Up'], index=recent_features.index)
                    ml_predictions['Predicted_Gap'] = np.argmax(pred_probs, axis=1).map({0: 'None', 1: 'Down', 2: 'Up'})
        else:
            st.warning("数据不足，无法训练ML模型。")

    # 新增：缺口交易策略回测（改进信号准确度）
    initial_capital = 10000.0
    trades = []
    if enable_strategy:
        # 初始化策略列
        data['Strategy_Signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
        data['Position'] = 0  # 当前仓位: 1: 多头, -1: 空头, 0: 无仓
        data['Entry_Price'] = np.nan
        data['Exit_Price'] = np.nan
        data['Strategy_Return'] = 0.0

        capital = initial_capital
        position = 0
        entry_price = 0.0
        gap_type_pos = None
        fill_target = 0.0
        equity = [initial_capital] * len(data)

        for i in range(len(data)):
            if i == 0:
                data.iloc[0, data.columns.get_loc('Position')] = 0
                data.iloc[0, data.columns.get_loc('Strategy_Signal')] = 0
                equity[0] = capital
                continue

            current_date = data.index[i]
            row = data.iloc[i]
            prev_date = data.index[i-1]
            
            # 生成信号（基础逻辑）
            base_signal = 0
            if row['Has_Gap']:
                if strategy_type == "简单缺口填补":
                    # 填补策略: Up Gap 做空（期待填补），Down Gap 做多
                    if row['Gap_Type'] == 'Up':
                        base_signal = -1  # 卖出（空头）
                    elif row['Gap_Type'] == 'Down':
                        base_signal = 1   # 买入（多头）
                elif strategy_type == "缺口延续":
                    # 延续策略: Up Gap 做多，Down Gap 做空
                    if row['Gap_Type'] == 'Up':
                        base_signal = 1   # 买入
                    elif row['Gap_Type'] == 'Down':
                        base_signal = -1  # 卖出
            
            # 改进：准确度过滤
            signal = 0
            if base_signal != 0:
                # 1. 成交量过滤
                volume_confirm = row['Volume'] > row['Volume_MA'] * volume_multiplier
                
                # 2. ML确认（如果启用）
                ml_confirm = True
                if enable_ml and ml_model and scaler:
                    current_features = scaler.transform(pd.DataFrame([row[features]]))
                    current_tensor = torch.tensor(current_features, dtype=torch.float32)
                    with torch.no_grad():
                        pred_output = ml_model(current_tensor)
                        pred_prob = torch.softmax(pred_output, dim=1).numpy()[0]
                        if base_signal == 1:  # 买入（期待Up或Down填补）
                            ml_prob = pred_prob[2] if row['Gap_Type'] == 'Up' else pred_prob[1]  # Up prob for continuation, Down for fill
                        else:  # 卖出
                            ml_prob = pred_prob[1] if row['Gap_Type'] == 'Up' else pred_prob[2]
                        ml_confirm = ml_prob > (ml_threshold / 100)
                
                # 3. ADX趋势确认（针对延续策略）
                adx_confirm = True
                if strategy_type == "缺口延续" and row['ADX'] < adx_threshold:
                    adx_confirm = False  # 弱趋势不触发延续信号
                
                # 组合过滤
                if volume_confirm and ml_confirm and adx_confirm:
                    signal = base_signal
            
            data.iloc[i, data.columns.get_loc('Strategy_Signal')] = signal
            
            exit_signal = False
            exit_reason = ''
            
            if signal != 0 and position == 0:
                # 开仓
                position = signal
                entry_price = row['Open']
                gap_type_pos = row['Gap_Type']
                fill_target = row['Prev_Close']
                data.iloc[i, data.columns.get_loc('Entry_Price')] = entry_price
                data.iloc[i, data.columns.get_loc('Position')] = position
                trades.append({
                    'date': current_date, 
                    'action': 'entry', 
                    'price': entry_price, 
                    'type': row['Gap_Type'],
                    'gap_size': abs(row['Gap_Size']),
                    'reason': f"Volume x{volume_multiplier}, ML {ml_prob*100:.0f}%, ADX {row['ADX']:.1f}"
                })
            
            elif position != 0:
                # 检查平仓条件: 缺口关闭 或 止损（动态止损基于波动率）
                dynamic_sl = stop_loss_pct / 100 * row['Volatility'] * np.sqrt(252) if row['Volatility'] > 0 else stop_loss_pct / 100  # 年化波动调整
                pnl_pct = ((row['Open'] - entry_price) / entry_price) * position
                if dynamic_sl > 0 and pnl_pct <= -dynamic_sl:
                    exit_signal = True
                    exit_reason = 'Dynamic Stop Loss'
                
                # 检查缺口填充（基于当日数据）
                if not exit_signal:  # 如果未触发止损，再检查填充
                    if gap_type_pos == 'Up':
                        partial_cond = row['Low'] <= fill_target
                        full_cond = row['Close'] <= fill_target
                    else:  # Down
                        partial_cond = row['High'] >= fill_target
                        full_cond = row['Close'] >= fill_target
                    
                    if full_cond:
                        exit_signal = True
                        exit_reason = 'Full Close'
                    elif partial_cond:
                        exit_signal = True
                        exit_reason = 'Partial Close'
                
                if exit_signal:
                    # 平仓（使用收盘价）
                    exit_price = row['Close']
                    data.iloc[i, data.columns.get_loc('Exit_Price')] = exit_price
                    data.iloc[i, data.columns.get_loc('Position')] = 0
                    
                    # 计算回报
                    trade_return = ((exit_price - entry_price) / entry_price) * position * (position_size / 100)
                    data.iloc[i, data.columns.get_loc('Strategy_Return')] = trade_return
                    capital *= (1 + trade_return)
                    
                    trades.append({
                        'date': current_date, 
                        'action': 'exit', 
                        'price': exit_price, 
                        'pnl': trade_return,
                        'reason': exit_reason
                    })
                    position = 0
                    entry_price = 0
                    gap_type_pos = None
                    fill_target = 0.0
            
            else:
                # 无信号，保持仓位
                data.iloc[i, data.columns.get_loc('Position')] = position
            
            # 更新权益
            equity[i] = capital
        
        final_return = (capital / initial_capital - 1) * 100

        # 策略绩效统计
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        exit_trades = trades_df[trades_df['action'] == 'exit']
        num_trades = len(exit_trades)
        win_rate = (exit_trades['pnl'] > 0).sum() / num_trades if num_trades > 0 else 0

    # 可视化 - 主图: 价格缺口 + 策略信号 + 权益曲线
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

        # 添加权益曲线到子图
        fig.add_trace(go.Scatter(x=data.index, y=equity, mode='lines', name='策略权益',
                                 line=dict(color='blue')), row=2, col=1)
        bh_equity = data['Close'] / data['Close'].iloc[0] * initial_capital
        fig.add_trace(go.Scatter(x=data.index, y=bh_equity, mode='lines', name='买入并持有',
                                 line=dict(color='orange')), row=2, col=1)

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
        borderwidth=1
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
        borderwidth=1
    ))

    # 更新布局
    fig.update_layout(yaxis_title='价格 (USD)', 
                      yaxis2_title='权益 (USD)',
                      xaxis_title='日期', 
                      xaxis2_title='日期',
                      title=f"{ticker} 价格缺口分析 ({period})",
                      height=800, showlegend=True,
                      hovermode='x unified',
                      annotations=annotations)
    fig.update_xaxes(rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # 新增：ML预测可视化
    if enable_ml and ml_predictions is not None:
        st.subheader("ML缺口预测（未来5天概率）")
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Bar(x=ml_predictions.index, y=ml_predictions['Up'], name='上缺口概率', marker_color='green'))
        fig_ml.add_trace(go.Bar(x=ml_predictions.index, y=ml_predictions['Down'], name='下缺口概率', marker_color='red'))
        fig_ml.add_trace(go.Scatter(x=ml_predictions.index, y=ml_predictions['None'], mode='lines', name='无缺口概率', line=dict(color='gray')))
        fig_ml.update_layout(title=f"{ticker} 未来缺口预测概率", xaxis_title='日期', yaxis_title='概率', barmode='stack')
        st.plotly_chart(fig_ml, use_container_width=True)

    # 策略绩效图（如果启用）
    if enable_strategy:
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

# 辅助函数：RSI计算
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
