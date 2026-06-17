Model 26.6.1
[first release]

<img width="1915" height="900" alt="Screenshot 2026-06-17 171614" src="https://github.com/user-attachments/assets/744a6b27-2c80-42ec-a7c1-66e32d2dd0e7" />

1. What is the model about?
This is an interactive Linear Regression model designed to predict property prices based on their physical size (Area). It serves as a visual learning lab to understand how Machine Learning models "learn" from data by finding the mathematical line of best fit.

2. How it works
The model takes in historical training data (Area vs. Price). It starts with a random guess for the "Weight" (price per sqft) and "Bias" (base price). Using an algorithm called Gradient Descent, it iteratively adjusts these parameters to minimize the difference (error) between its predictions and the actual data.

3. Technology & Tools Used
Frontend / UI: Streamlit (Python web framework)
UI/UX Aesthetics: Custom CSS3 & Vanilla JS (Glassmorphism, Apple-style scroll reveals, Liquid mesh gradients, Google Material ripples)
Data Processing: NumPy (for high-speed array math) & Pandas (for dataframes)
Visualizations: Matplotlib (for generating 2D Regression, 2D Contour, and 3D Surface plots)

4. Maths & Scientific Logic UsedLinear Equation: $f_{w,b}(x) = wx + b$Mean Squared Error (Cost Function): $J(w,b) = \frac{1}{2m} \sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})^2$Gradient Descent Optimization: Calculates the partial derivatives $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$ to mathematically step "downhill" towards the lowest possible error.
  
5. Benefits
Provides real-time, instantaneous visual feedback on abstract mathematical concepts.
Eliminates the "black box" nature of AI by exposing the exact calculations taking place under the hood.
Allows manual experimentation to see why the optimized model is mathematically superior to human guessing.
6. Target Audience
Students & Beginners: Anyone learning Data Science, Machine Learning, or AI for the first time.
Educators & Instructors: Teachers who need an interactive, highly-visual aid to explain Gradient Descent and Cost Functions.
Developers: Software engineers looking to understand the underlying math of AI before moving on to complex neural networks.

<img width="1460" height="938" alt="image" src="https://github.com/user-attachments/assets/b2363d4f-4c4a-4840-8fd9-7e40630c265c" />


