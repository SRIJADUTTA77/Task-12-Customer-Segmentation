# 📊 Task-12: Mall Customer Segmentation using Machine Learning

## 🔹 Project Overview
This project implements an **unsupervised machine learning pipeline** to perform customer segmentation using **K-Means Clustering**. The objective is to identify distinct customer groups based on purchasing behavior, enabling data-driven marketing strategies, personalized targeting, and improved business decision-making.

Given that customer segmentation does not have labeled outcomes, this study emphasizes exploratory data analysis, feature scaling, and clustering evaluation through visual and statistical techniques.

---

## 📁 Project Structure

Task-12-Customer-Segmentation/
│── dataset/
│ └── mall_customers_task12.csv
│
│── src/
│ └── customer_segmentation_kmeans.py
│
│── outputs/
│ ├── elbow_plot.png
│ ├── clusters_scatter.png
│ └── cluster_summary.csv
│
│── README.md
│── .gitignore


---

## 📌 Dataset Information

**Source:** Custom dataset aligned with standard Mall Customers dataset format  

**Records:** 200 customers  
**Features:** 5 variables  

| Column | Description |
|--------|-------------|
| `CustomerID` | Unique identifier |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Annual Income (k$)` | Annual income in thousands |
| `Spending Score (1-100)` | Customer spending behavior score |

**Target Objective:** Discover meaningful customer segments based on:
- Annual Income  
- Spending Score  

---

## 🛠 Technologies Used

- **Python 3.11**
- **Pandas** — Data manipulation  
- **NumPy** — Numerical computations  
- **Matplotlib** — Data visualization  
- **Scikit-learn** — Machine Learning (K-Means Clustering)

---

## 🤖 Machine Learning Approach

### 🔹 Step 1 — Data Preprocessing
- Selected key features:  
  - Annual Income  
  - Spending Score  
- Applied **StandardScaler** for normalization to ensure fair clustering.

### 🔹 Step 2 — Optimal Cluster Selection (Elbow Method)
- Evaluated inertia values for k = 1 to 10  
- Identified **k = 5** as the optimal number of clusters  
- Saved visualization: `outputs/elbow_plot.png`

### 🔹 Step 3 — K-Means Clustering (k = 5)
- Applied K-Means algorithm  
- Assigned each customer to one of five clusters  
- Generated scatter visualization: `outputs/clusters_scatter.png`

### 🔹 Step 4 — Cluster Profiling
- Computed mean values per cluster  
- Saved statistical summary: `outputs/cluster_summary.csv`

---

## 📈 Key Business Insights

Based on typical clustering patterns, customers can be interpreted as:

| Cluster | Profile |
|--------|---------|
| 0 | Low income, low spending |
| 1 | High income, high spending (Premium customers) |
| 2 | Medium income, medium spending |
| 3 | Low income, high spending (Potential loyal customers) |
| 4 | High income, low spending (Target for promotions) |

💡 **Business Implication:**  
These insights enable:
- Personalized marketing  
- Loyalty programs  
- Budget allocation optimization  
- Strategic customer engagement  

---

## ▶️ How to Run the Project

### Step 1 — Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
Step 2 — Execute the Script
cd src
python customer_segmentation_kmeans.py
Step 3 — Verify Outputs
Check the outputs/ folder for:

elbow_plot.png

clusters_scatter.png

cluster_summary.csv

📌 Key Learnings
Clustering is an exploratory technique — labels are inferred, not predefined.

Feature scaling is critical for distance-based algorithms like K-Means.

The Elbow Method is an effective heuristic for selecting optimal k.

Visual analysis significantly enhances interpretability of results.

🎯 Conclusion
This project demonstrates a complete end-to-end unsupervised learning workflow for customer segmentation, integrating data preprocessing, clustering, visualization, and business interpretation. It provides a strong foundation for advanced customer analytics and strategic decision-making.

👨‍💻 Author
Srija Dutta
Task-12 — Machine Learning & AI

