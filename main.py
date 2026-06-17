import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------
# MODEL FUNCTIONS
# -----------------

def compute_cost(x, y, w, b):
    m = x.shape[0] 
    if m == 0: return 0
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    total_cost = (1 / (2 * m)) * cost
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J_history.append(compute_cost(x, y, w, b))
    return w, b, J_history

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

# -----------------
# STREAMLIT UI
# -----------------

st.set_page_config(layout="wide")

# --- UI/UX INJECTION ---
try:
    with open("static/css/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Inject JS via component (runs in iframe, targets parent)
    import streamlit.components.v1 as components
    with open("static/js/script.js") as f:
        js_code = f.read()
        components.html(f"<script>{js_code}</script>", height=0, width=0)
except Exception as e:
    st.warning(f"Could not load UI/UX assets: {e}")
# -----------------------

st.title("Property Analysis by ML (Interactive Learning Lab)")
st.write("Train a Linear Regression model, experiment with custom parameters, and visualize the Cost Function J(w,b) in 2D and 3D.")

with st.expander("📖 ReadMe"):
    st.markdown("""
    ### 1. What is the model about?
    This is an interactive **Linear Regression** model designed to predict property prices based on their physical size (Area). It serves as a visual learning lab to understand how Machine Learning models "learn" from data by finding the mathematical line of best fit.
    
    ### 2. How it works
    The model takes in historical training data (Area vs. Price). It starts with a random guess for the "Weight" (price per sqft) and "Bias" (base price). Using an algorithm called Gradient Descent, it iteratively adjusts these parameters to minimize the difference (error) between its predictions and the actual data.
    
    ### 3. Technology & Tools Used
    - **Frontend / UI:** Streamlit (Python web framework)
    - **UI/UX Aesthetics:** Custom CSS3 & Vanilla JS (Glassmorphism, Apple-style scroll reveals, Liquid mesh gradients, Google Material ripples)
    - **Data Processing:** NumPy (for high-speed array math) & Pandas (for dataframes)
    - **Visualizations:** Matplotlib (for generating 2D Regression, 2D Contour, and 3D Surface plots)
    
    ### 4. Maths & Scientific Logic Used
    - **Linear Equation:** $f_{w,b}(x) = wx + b$
    - **Mean Squared Error (Cost Function):** $J(w,b) = \\frac{1}{2m} \\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$
    - **Gradient Descent Optimization:** Calculates the partial derivatives $\\frac{\\partial J}{\\partial w}$ and $\\frac{\\partial J}{\\partial b}$ to mathematically step "downhill" towards the lowest possible error.
    
    ### 5. Benefits
    - Provides real-time, instantaneous visual feedback on abstract mathematical concepts.
    - Eliminates the "black box" nature of AI by exposing the exact calculations taking place under the hood.
    - Allows manual experimentation to see *why* the optimized model is mathematically superior to human guessing.
    
    ### 6. Target Audience
    - **Students & Beginners:** Anyone learning Data Science, Machine Learning, or AI for the first time.
    - **Educators & Instructors:** Teachers who need an interactive, highly-visual aid to explain Gradient Descent and Cost Functions.
    - **Developers:** Software engineers looking to understand the underlying math of AI before moving on to complex neural networks.
    """)

# Initialize Data state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
        "Area (1000 sqft)": [1.0, 1.5, 2.0, 2.5],
        "Price (1000s $)": [300.0, 420.0, 500.0, 580.0]
    })

# Custom lines state
if 'custom_wb' not in st.session_state:
    st.session_state.custom_wb = pd.DataFrame({
        "Weight (w)": [150.0, 250.0],
        "Bias (b)": [50.0, 100.0]
    })

if 'w' not in st.session_state: st.session_state.w = 180.0
if 'b' not in st.session_state: st.session_state.b = 130.0

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Training Data (Actual)")
    edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", hide_index=True)
    st.session_state.df = edited_df
    x_train = edited_df["Area (1000 sqft)"].values
    y_train = edited_df["Price (1000s $)"].values
    
    with st.expander("🧠 Thinking: Training Data ($x$, $y$) Details"):
        st.markdown("""
        **What is this?**
        This table holds your $m$ training examples.
        - **$x^{(i)}$ (Features):** The independent variable (Area).
        - **$y^{(i)}$ (Targets):** The actual "Ground Truth" answers (Price).
        
        **How the code works:**
        The model dynamically reads this table into highly efficient **NumPy Arrays** (`x_train`, `y_train`). No matter how many rows you add, NumPy processes them instantly using vectorized loops.
        """)

with col2:
    st.subheader("2. Custom Model Parameters")
    st.write("Add multiple w,b pairs to plot multiple lines on the graphs.")
    edited_wb = st.data_editor(st.session_state.custom_wb, num_rows="dynamic", hide_index=True)
    st.session_state.custom_wb = edited_wb

    with st.expander("🧠 Thinking: Model Parameters ($w$, $b$)"):
        st.markdown("""
        **What is this?**
        The equation of a straight line is $f_{w,b}(x) = wx + b$.
        - **Weight ($w$)**: The slope of the line. In this context, it represents the *price per square foot*.
        - **Bias ($b$)**: The y-intercept. It represents the *base price* before area is factored in.
        
        **Experimentation:**
        By manually changing $w$ and $b$ here, you can see how "bad" guesses look visually compared to the mathematically optimized "Trained" line.
        """)

st.divider()

col_train, col_pred = st.columns(2)
with col_train:
    st.subheader("3. Gradient Descent Training")
    alpha = st.number_input("Learning Rate (alpha)", value=0.01, format="%.4f")
    iterations = st.number_input("Iterations", min_value=1, value=1000, step=100)
    
    if st.button("Run Gradient Descent"):
        with st.spinner("Training model..."):
            w_final, b_final, J_hist = gradient_descent(x_train, y_train, w_in=0.0, b_in=0.0, alpha=alpha, num_iters=iterations)
            st.session_state.w = w_final
            st.session_state.b = b_final
            st.success(f"Training Complete! Minimized Cost J: **{J_hist[-1]:.2f}**")
            
    current_cost = compute_cost(x_train, y_train, st.session_state.w, st.session_state.b)
    st.info(f"**Trained Model:** $f_{{w,b}}(x) = {st.session_state.w:.2f}x + {st.session_state.b:.2f}$ \n\n **Cost $J$:** `{current_cost:,.2f}`")

    with st.expander("🧠 Thinking: Gradient Descent & Cost Function Algorithm"):
        st.markdown("""
        **The Cost Function $J(w,b)$**
        Calculates how "wrong" the current line is by squaring the errors:
        $$J(w,b) = \\frac{1}{2m} \\sum_{i=1}^{m} \\left( f_{w,b}(x^{(i)}) - y^{(i)} \\right)^2$$
        
        **Gradient Descent Algorithm**
        To find the minimum cost, the model calculates the derivative (slope) of the cost function, then takes a tiny step downhill.
        - $w = w - \\alpha \\frac{\\partial J}{\\partial w}$
        - $b = b - \\alpha \\frac{\\partial J}{\\partial b}$
        
        **Variables:**
        - **$\\alpha$ (Learning Rate)**: Controls how big of a step to take. Too small = slow. Too big = it overshoots and breaks!
        - **Iterations**: How many times we repeat the loop to reach the bottom of the error bowl.
        """)

with col_pred:
    st.subheader("4. Predict Property Price")
    pred_area = st.number_input("Enter a property Area (in 1000 sqft):", min_value=0.1, value=2.0, step=0.1)
    pred_price = st.session_state.w * pred_area + st.session_state.b
    st.success(f"Predicted price: **${pred_price*1000:,.2f}**")

    with st.expander("🧠 Thinking: Model Inference"):
        st.markdown("""
        **What is Inference?**
        Once Gradient Descent has successfully trained the model (found the optimal $w$ and $b$), we can do "Inference".
        
        **The Math:**
        We simply plug your new hypothetical Area ($x$) into the final learned equation:
        $f_{w,b}(\\text{Area}) = (w \\times \\text{Area}) + b$
        
        This generates the predicted $\\hat{y}$ (Price) shown above.
        """)

st.divider()

# GRAPHING SECTION
st.header("Visualizations")
st.write("Explore the relationship between the linear regression lines and the Cost Function J(w,b) landscape.")

tab1, tab2, tab3 = st.tabs(["2D Linear Regression Plot", "2D Cost Contour Plot", "3D Cost Surface (Soup Bowl)"])

with tab1:
    st.subheader("Linear Regression with Multiple Custom Lines")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Actual data
    ax1.scatter(x_train, y_train, marker='x', color='red', s=100, label='Actual Data')
    
    if len(x_train) > 0:
        x_line = np.linspace(min(x_train)*0.8, max(x_train)*1.2, 100)
        
        # Plot trained line
        y_line = compute_model_output(x_line, st.session_state.w, st.session_state.b)
        ax1.plot(x_line, y_line, color='blue', linewidth=3, label=f'Trained Line (w={st.session_state.w:.1f}, b={st.session_state.b:.1f})')
        
        # Plot custom lines
        colors = plt.cm.rainbow(np.linspace(0, 1, len(edited_wb)))
        for idx, row in edited_wb.iterrows():
            cw = row["Weight (w)"]
            cb = row["Bias (b)"]
            cy_line = compute_model_output(x_line, cw, cb)
            c_cost = compute_cost(x_train, y_train, cw, cb)
            ax1.plot(x_line, cy_line, color=colors[idx], linestyle='--', label=f'Custom {idx+1} (w={cw}, b={cb}) Cost={c_cost:.0f}')

        # Plot user's prediction
        ax1.scatter(pred_area, pred_price, marker='o', color='green', s=150, zorder=5, label='Prediction')
        
    ax1.set_title("Property Pricing Prediction")
    ax1.set_ylabel("Price (in 1000s of dollars)")
    ax1.set_xlabel("Size (1000 sqft)")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    with st.expander("🧠 Thinking: The 2D Regression Graph"):
        st.markdown("""
        **Visualizing the Error:**
        Look at the vertical distance between the red 'x' marks (actual data) and the blue line (our prediction). That invisible vertical line is the **Error**. 
        
        The Cost Function takes every single one of those vertical distances, squares them, and adds them up. Notice how your Custom Lines (dashed) might miss the red X's entirely? That results in a massive Cost penalty!
        """)

# Generate Meshgrid for Cost plots
if len(x_train) > 0:
    w_range = np.linspace(st.session_state.w - 200, st.session_state.w + 200, 40)
    b_range = np.linspace(st.session_state.b - 200, st.session_state.b + 200, 40)
    W, B = np.meshgrid(w_range, b_range)
    J = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i,j] = compute_cost(x_train, y_train, W[i,j], B[i,j])
            
    with tab2:
        st.subheader("2D Contour Plot of Cost Function $J(w,b)$")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        contour = ax2.contour(W, B, J, levels=30, cmap='viridis')
        ax2.clabel(contour, inline=True, fontsize=8)
        
        # Plot custom points
        colors = plt.cm.rainbow(np.linspace(0, 1, len(edited_wb)))
        for idx, row in edited_wb.iterrows():
            cw = row["Weight (w)"]
            cb = row["Bias (b)"]
            ax2.scatter(cw, cb, color=colors[idx], marker='x', s=100, label=f'Custom {idx+1}')
            
        # Plot trained point
        ax2.scatter(st.session_state.w, st.session_state.b, color='blue', marker='*', s=200, label='Trained Minimum')
        
        ax2.set_xlabel('Weight (w)')
        ax2.set_ylabel('Bias (b)')
        ax2.set_title('Cost Function Contour Map')
        ax2.legend()
        st.pyplot(fig2)

        with st.expander("🧠 Thinking: The Topographical Contour Map"):
            st.markdown("""
            **Reading the Map:**
            Imagine you are looking straight down from a helicopter at a valley. Each ring represents a specific "elevation" of Error (Cost). 
            
            - The very center of the smallest ring is the **Global Minimum**—the absolute lowest possible error for this dataset.
            - Notice how the blue star (your trained model) sits perfectly in the center? 
            - Notice how your custom 'x' marks sit on outer rings? They have a higher "elevation" of error!
            """)

    with tab3:
        st.subheader("3D Surface Plot (The Soup Bowl)")
        fig3 = plt.figure(figsize=(10, 8))
        ax3 = fig3.add_subplot(111, projection='3d')
        
        surf = ax3.plot_surface(W, B, J, cmap='viridis', alpha=0.8, edgecolor='none')
        
        ax3.set_xlabel('Weight (w)')
        ax3.set_ylabel('Bias (b)')
        ax3.set_zlabel('Cost J(w,b)')
        ax3.set_title('3D View of the Cost Function')
        fig3.colorbar(surf, shrink=0.5, aspect=5)
        st.pyplot(fig3)

        with st.expander("🧠 Thinking: The 3D Soup Bowl Structure"):
            st.markdown("""
            **Why does Gradient Descent always work here?**
            This 3D plot visualizes the Squared Error Cost Function $J(w,b)$. Notice its shape? It looks like a soup bowl!
            
            In mathematics, this is called a **Convex Function**. A convex function has one, and exactly one, minimum point at the very bottom of the bowl. It has no hidden valleys or false bottoms. 
            
            This is why Gradient Descent is so powerful for Linear Regression: no matter where you start on the edges of the bowl (your initial $w$ and $b$ guesses), taking a mathematical step "downhill" will **always** eventually lead you to the exact bottom!
            """)
