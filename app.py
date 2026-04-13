import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

st.set_page_config(page_title="ML Visualizer", layout="wide")

st.title("📊 ML Foundations Visualizer")

menu = st.sidebar.selectbox("Select Module", [
    "Vector Similarity",
    "Function & Derivative",
    "Gradient Descent"
])

# ================= VECTOR =================
if menu == "Vector Similarity":
    v1 = np.array(st.text_input("Vector 1", "1,2").split(","), dtype=float)
    v2 = np.array(st.text_input("Vector 2", "3,4").split(","), dtype=float)

    dot = np.dot(v1, v2)
    cos = dot / (np.linalg.norm(v1)*np.linalg.norm(v2))
    dist = np.linalg.norm(v1 - v2)
    angle = np.degrees(np.arccos(cos))

    st.write("Dot:", dot)
    st.write("Cosine:", cos)
    st.write("Distance:", dist)
    st.write("Angle:", angle)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,v1[0]], y=[0,v1[1]], mode='lines', name="v1"))
    fig.add_trace(go.Scatter(x=[0,v2[0]], y=[0,v2[1]], mode='lines', name="v2"))
    st.plotly_chart(fig)

# ================= FUNCTION =================
elif menu == "Function & Derivative":
    x = sp.symbols('x')
    expr = st.text_input("Function", "x**2")

    f = sp.sympify(expr)
    df = sp.diff(f, x)

    st.write("Derivative:", df)

    f_np = sp.lambdify(x, f, "numpy")
    df_np = sp.lambdify(x, df, "numpy")

    x_vals = np.linspace(-10,10,100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=f_np(x_vals), name="f(x)"))
    fig.add_trace(go.Scatter(x=x_vals, y=df_np(x_vals), name="f'(x)"))
    st.plotly_chart(fig)

# ================= GRADIENT =================
elif menu == "Gradient Descent":
    lr = st.slider("Learning Rate", 0.01, 1.0, 0.1)
    it = st.slider("Iterations", 1, 50, 20)

    def f(x): return x**2
    def g(x): return 2*x

    x = 5
    loss = []

    for i in range(it):
        x = x - lr*g(x)
        loss.append(f(x))

    st.write("Final value:", x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss, mode='lines+markers'))
    st.plotly_chart(fig)