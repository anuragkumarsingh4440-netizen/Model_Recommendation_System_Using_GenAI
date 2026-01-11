# 🤖 GenAI Model Recommendation System  
### Model Selection + Statistical Intelligence for AI/ML Engineers
STREAMLIT LINK :- https://modelstats.streamlit.app/
LINKED IN :- https://www.linkedin.com/in/anurag-kumar-singh4440?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BzOav3YVtTO%2BSTfBbUTZzjw%3D%3D
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/92992c74-c86c-46de-8af1-aa97825e4520" />
---
![Uploading image.png…]()


---

> ⚡ Stop guessing models.  
> 📊 Let data statistics + GenAI guide your decisions.  
> 🧠 Built for real-world AI/ML workflows — not toy datasets.

---

## 🚀 Project Overview

Choosing the *right* machine learning model is one of the **hardest and most time-consuming** steps in any AI/ML project.

Most engineers:
- Train multiple models blindly  
- Spend days tuning  
- Realize **late** which model actually fits the data  

This project was built to **break that loop**.

### 💡 What this system does
- Profiles your dataset statistically  
- Understands the **problem type automatically**  
- Uses **Generative AI reasoning** to recommend:
  - ✅ Best primary model  
  - 🔁 Strong backup model  
  - ❌ Why other models are not ideal  
  - 🧠 When neural networks *should* or *shouldn’t* be used  
- Gives **actionable EDA decisions**, not just numbers

---

## 🧠 Why This Project Exists (Real Struggle)

I faced this problem repeatedly while working on ML projects:

- Same dataset → different models → different results  
- No clear rule *why* one model worked better  
- Statistical checks done manually, scattered across notebooks  
- Model decisions based more on **habit** than **data evidence**

So instead of writing *another notebook*, I decided to build:

> **A decision system, not a training script**

This project combines:
- 📊 Statistical reasoning  
- 🤖 GenAI reasoning  
- 🎨 Clean, interactive UI  
- ⚙️ Practical ML engineering logic  

---

## 🧩 Core Features

### 🏆 Intelligent Model Recommendation
- Primary model (with accuracy range)  
- Backup model (with accuracy range)  
- Clear comparison table  
- Human-readable reasoning  

### 🧠 Explainability (Not Black Box)
- Why this model fits the data  
- Why other common models fail  
- Neural network recommendation with justification  

### 📊 Statistical Analysis Toolkit
- Mean, Median, Variance, Std Dev  
- Skewness & Kurtosis  
- IQR-based outlier detection  

### 🛠️ Dynamic EDA Action Engine
Instead of just showing stats, the system tells you **what to do next**:

| Data Condition        | Suggested Action                     |
|-----------------------|--------------------------------------|
| High missing values   | KNN / MICE or feature drop            |
| Heavy outliers        | Log transform or robust models        |
| High skewness         | Box-Cox / log transform               |
| Heavy tails           | RobustScaler                          |

---

## ⚙️ How It Works (High-Level Flow)

```text
CSV Dataset
     ↓
Statistical Profiling
     ↓
Problem Type Detection
     ↓
GenAI Reasoning Engine
     ↓
Model Recommendation + Explanations
     ↓
EDA Actions & Report Export
