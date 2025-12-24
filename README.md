# 📊 Product Analysis & Sentiment Dashboard

This project is a comprehensive Data Intelligence dashboard built with **Streamlit**. It provides a 360-degree view of product performance, customer testimonials, and deep-learning-based sentiment analysis of user reviews.

## 🎯 Purpose
The goal of this application is to transform raw scraped data into actionable business insights. It allows stakeholders to:
1. **Monitor Inventory:** View cleaned product lists and pricing.
2. **Analyze Feedback:** Evaluate customer testimonials and ratings.
3. **Understand Sentiment:** Use Deep Learning (NLP) to detect the emotional tone of reviews and identify key topics via Word Clouds and temporal trends.

---

## 🚀 Key Features

### 1. Data Management & Cleaning
* **Automated Processing:** Loads data from pre-scraped CSV files (`products`, `reviews`, `testimonials`).
* **Standardization:** Prices are formatted with currency symbols, and dates are standardized to `DD/MM/YYYY`.
* **Clean UI:** All raw metadata (like `price_raw` or empty date columns) is removed before display.

### 2. Advanced Reviews Analytics
* **Dynamic Temporal Filtering:** A custom-built bi-directional filter allows users to select data by specific years (Radio buttons) or specific month ranges (Slider).
* **Deep Learning Sentiment Analysis:** Uses the `distilbert-base-uncased-finetuned-sst-2-english` model to classify reviews as **POSITIVE** or **NEGATIVE** with high confidence scores.
* **Sentiment Trends:** A time-series line chart tracks how customer satisfaction evolves month-over-month.
* **Word Clouds:** Visually identifies the most frequent keywords in both positive and negative feedback groups.

### 3. Performance & Deployment
* **Caching:** Optimized with `@st.cache_data` and `@st.cache_resource` to minimize load times and memory usage on the Render free tier.
* **Mobile-Friendly:** Responsive layout using Streamlit’s wide-mode grid system.

---

## 🛠 Tech Stack
* **Language:** Python 3.10+
* **Framework:** Streamlit
* **Data Science:** Pandas, NumPy
* **NLP/AI:** HuggingFace Transformers (DistilBERT), PyTorch
* **Visualization:** Plotly Express, WordCloud, Matplotlib

---

## 📂 Project Structure
```text
├── app.py                # Main Streamlit application code
├── requirements.txt      # Python dependencies
├── products_data.csv     # Scraped product information
├── reviews_data.csv      # Customer reviews and star ratings
└── testimonials_data.csv # Short-form customer testimonials
