from graphviz import Digraph

# 创建有向图
dot = Digraph(name="MyLSTM", format="svg", engine="dot")
dot.attr(rankdir="TB", splines="ortho")  # 从上到下布局，直角连接线

# ------------------------------
# 定义 LSTM 单元内部结构（子图）
# ------------------------------
with dot.subgraph(name="cluster_LSTM_Cell") as c:
    c.attr(label="LSTM Cell (PyTorch内置实现)", color="lightgrey", style="rounded")
    
    # 输入部分
    c.node("x_t", "输入 x_t\n(batch, feature)", shape="box")
    c.node("h_t-1", "短时记忆输入 h_{t-1}", shape="box")
    c.node("C_t-1", "长时记忆输入 C_{t-1}", shape="box")
    
    # 门控结构
    c.node("forget_gate", "遗忘门 σ", shape="ellipse")
    c.node("input_gate", "输入门 σ", shape="ellipse")
    c.node("cell_gate", "候选记忆 tanh", shape="ellipse")
    c.node("output_gate", "输出门 σ", shape="ellipse")
    
    # 输出部分
    c.node("C_t", "长时记忆输出 C_t", shape="box")
    c.node("h_t", "短时记忆输出 h_t", shape="box")
    
    # 连接关系
    c.edge("h_t-1", "forget_gate", label="W_f")
    c.edge("x_t", "forget_gate", label="W_f")
    c.edge("h_t-1", "input_gate", label="W_i")
    c.edge("x_t", "input_gate", label="W_i")
    c.edge("h_t-1", "cell_gate", label="W_c")
    c.edge("x_t", "cell_gate", label="W_c")
    c.edge("h_t-1", "output_gate", label="W_o")
    c.edge("x_t", "output_gate", label="W_o")
    
    c.edge("forget_gate", "C_t", label="⊗")
    c.edge("C_t-1", "C_t", label="⊗")
    c.edge("input_gate", "C_t", label="⊗")
    c.edge("cell_gate", "C_t", label="⊗")
    c.edge("C_t", "h_t", label="tanh")
    c.edge("output_gate", "h_t", label="⊗")

# ------------------------------
# 定义模型高层架构（你的代码）
# ------------------------------
dot.node("input", "输入\n(batch, seq, input_size)", shape="box")
dot.node("lstm", "LSTM Layer\n(num_layers=2, hidden_size=hidden_size)", shape="box")
dot.node("avg", "序列平均\n(dim=1)", shape="box")
dot.node("dropout", "Dropout\n(p=drop_prob)", shape="box")
dot.node("fc", "全连接层\n(hidden_size → output_size)", shape="box")
dot.node("tanh", "tanh 激活\n(输出)", shape="box")

# 连接高层架构
dot.edge("input", "lstm")
dot.edge("lstm", "avg")
dot.edge("avg", "dropout")
dot.edge("dropout", "fc")
dot.edge("fc", "tanh")

# 连接 LSTM 单元与高层架构
dot.edge("lstm", "h_t-1", style="dashed", color="blue", label="隐藏状态传递")
dot.edge("lstm", "C_t-1", style="dashed", color="blue")

# 保存为 SVG
dot.render("lstm_structure", view=True)