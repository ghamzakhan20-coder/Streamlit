import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---- Page Config ----
st.set_page_config(
    page_title="Weather & Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
    /* Main background and styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6bbfff 0%, #5aaeee 50%, #4569c4 100%);
        box-shadow: 4px 0 10px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: bold;
    }
    
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Card-like containers */
    .stMetric {
        background: #6bbfff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 30px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        border: none;
        font-weight: bold;
    }
    
    /* Headers and titles */
    h1, h2, h3 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* Selectbox */
    .stSelectbox {
        background: white;
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border: 2px solid rgba(255, 255, 255, 0.3);
        margin: 2rem 0;
    }
    
    /* Container backgrounds */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Radio buttons in sidebar */
    .stRadio > label {
        color: white;
        font-weight: bold;
    }
    
    /* Mobile Responsive Styles */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        h1 {
            font-size: 2rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
        }
        
        h3 {
            font-size: 1.2rem !important;
        }
        
        .stMetric {
            padding: 15px;
            margin-bottom: 10px;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
        }
        
        .stButton>button {
            width: 100%;
            padding: 12px 20px;
            font-size: 0.9rem;
        }
        
        .stDownloadButton>button {
            width: 100%;
            padding: 12px 20px;
            font-size: 0.9rem;
        }
        
        /* Make charts responsive */
        .stpyplot {
            max-width: 100%;
            height: auto;
        }
        
        /* Adjust columns for mobile */
        [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        
        /* Sidebar adjustments */
        [data-testid="stSidebar"] {
            width: 100% !important;
        }
        
        /* Fix overflow on small screens */
        .dataframe {
            overflow-x: auto;
            display: block;
            max-width: 100%;
        }
        
        /* Expander adjustments */
        [data-testid="stExpander"] {
            padding: 10px;
        }
        
        /* Info boxes mobile friendly */
        .stAlert {
            font-size: 0.9rem;
            padding: 10px;
        }
    }
    
    /* Tablet Responsive */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main {
            padding: 1.5rem;
        }
        
        h1 {
            font-size: 2.5rem !important;
        }
        
        .stMetric {
            padding: 18px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ---- Animated Title ----
st.markdown("""
    <h1 style='text-align: center; font-size: clamp(2rem, 5vw, 3.5rem); margin-bottom: 1rem;'>
        🌦️ Weather & Health Dashboard
    </h1>
    <p style='text-align: center; color: white; font-size: clamp(0.9rem, 2vw, 1.2rem); margin-bottom: 2rem;'>
        Track environmental factors and their impact on public health
    </p>
""", unsafe_allow_html=True)

# ---- Load Data ----
@st.cache_data
def load_data():
    return pd.read_excel("Project Problem.xlsx")

df = load_data()

# ---- Sidebar Navigation with Icons ----
st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.markdown("---")
section = st.sidebar.radio(
    "Choose a section:",
    ["🏠 Dashboard", "📄 Dataset", "📊 Statistics", "📈 Visualizations", "🤖 ML Model", "💡 Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Info")
st.sidebar.info(f"📅 Total Records: {df.shape[0]}")
st.sidebar.info(f"📋 Total Features: {df.shape[1]}")

# ------------------ Dashboard ------------------
if section == "🏠 Dashboard":
    st.markdown("## 🏠 Key Metrics Overview")
    st.markdown("---")
    
    # Metrics in cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📅 Total Months", df.shape[0], delta="Complete Data")
    with col2:
        st.metric("🌧️ Max Rainfall", f"{df['Rainfall'].max()} mm", delta="Peak Value")
    with col3:
        st.metric("🏥 Max Hospital Visits", df['Hospital_Visits'].max(), delta="Critical")
    with col4:
        st.metric("💨 Avg AQI", round(df['AQI'].mean(), 2), delta="Air Quality")
    
    st.markdown("---")
    
    # Additional dashboard cards
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("### 🌡️ Temperature Range")
            temp_min = df['Temperature'].min()
            temp_max = df['Temperature'].max()
            st.write(f"**Min:** {temp_min}°C | **Max:** {temp_max}°C")
            st.progress(int((temp_max - temp_min) / temp_max * 100))
    
    with col2:
        with st.container():
            st.markdown("### 💧 Humidity Levels")
            hum_avg = df['Humidity'].mean()
            st.write(f"**Average:** {round(hum_avg, 2)}%")
            st.progress(int(hum_avg))

# ------------------ Dataset ------------------
elif section == "📄 Dataset":
    st.markdown("## 📄 Dataset Explorer")
    st.markdown("---")
    
    # Expandable sections
    with st.expander("🔍 View Dataset Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💾 Download the complete dataset in CSV format")
    with col2:
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            file_name="weather_health_dataset.csv",
            mime="text/csv"
        )

# ------------------ Descriptive Stats ------------------
elif section == "📊 Statistics":
    st.markdown("## 📊 Descriptive Statistics")
    st.markdown("---")
    
    with st.container():
        st.write(df.describe())
    
    st.markdown("")
    st.divider()
    
    # Additional stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"✅ **Mean Hospital Visits:** {round(df['Hospital_Visits'].mean(), 2)}")
    with col2:
        st.warning(f"⚠️ **Std Deviation AQI:** {round(df['AQI'].std(), 2)}")
    with col3:
        st.error(f"🔴 **Max Temperature:** {df['Temperature'].max()}°C")

# ------------------ Visualizations ------------------
elif section == "📈 Visualizations":
    st.markdown("## 📈 Data Visualizations")
    st.markdown("---")

    # Hospital Visits Over Time
    with st.container():
        st.markdown("### 🏥 Hospital Visits Trend")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df["Month"], df["Hospital_Visits"], marker='o', color='#667eea', linewidth=2.5, markersize=8)
        ax.set_xlabel("Month", fontsize=12, fontweight='bold')
        ax.set_ylabel("Hospital Visits", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f7fafc')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    st.divider()

    # Histogram & Boxplot in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💨 AQI Distribution")
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df["AQI"], bins=10, color='#f093fb', edgecolor='#764ba2', linewidth=1.5)
        ax.set_xlabel("AQI", fontweight='bold')
        ax.set_ylabel("Frequency", fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f7fafc')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("### 🌧️ Rainfall Distribution")
        fig, ax = plt.subplots(figsize=(8,5))
        bp = ax.boxplot(df["Rainfall"], patch_artist=True)
        bp['boxes'][0].set_facecolor('#667eea')
        bp['boxes'][0].set_alpha(0.7)
        ax.set_ylabel("Rainfall (mm)", fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f7fafc')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.divider()

    # Correlation Heatmap
    with st.container():
        st.markdown("### 🔥 Correlation Heatmap")
        corr = df[["Temperature","Rainfall","Humidity","AQI","Hospital_Visits"]].corr()
        fig, ax = plt.subplots(figsize=(10,8))
        cax = ax.matshow(corr, cmap='coolwarm')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='left')
        plt.yticks(range(len(corr.columns)), corr.columns)
        fig.colorbar(cax)
        
        # Add correlation values
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                       ha='center', va='center', color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ------------------ ML Model ------------------
elif section == "🤖 ML Model":
    st.markdown("## 🤖 Machine Learning Predictions")
    st.markdown("---")
    
    X = df[["Temperature", "Rainfall", "Humidity", "AQI"]]
    y = df["Hospital_Visits"]

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    st.success("✅ Model trained successfully using Linear Regression!")
    
    st.markdown("")
    
    # Model metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 R² Score", f"{r2:.3f}")
    with col2:
        st.metric("📉 MAE", f"{mae:.2f}")
    with col3:
        st.metric("🎯 Accuracy", f"{r2*100:.1f}%")

    st.divider()

    # Predicted vs Actual
    with st.container():
        st.markdown("### 📈 Predicted vs Actual Hospital Visits")
        pred_df = pd.DataFrame({"Actual": y, "Predicted": predictions})
        st.line_chart(pred_df, use_container_width=True)

    st.divider()

    # Feature visualization
    st.markdown("### 🔍 Feature Explorer")
    feature = st.selectbox("Select a feature to visualize", df.columns, index=0)
    
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df[feature], marker='o', color='#764ba2', linewidth=2)
    ax.set_xlabel("Index", fontweight='bold')
    ax.set_ylabel(feature, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f7fafc')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ------------------ Insights ------------------
elif section == "💡 Insights":
    st.markdown("## 💡 Key Insights & Findings")
    st.markdown("---")
    
    max_rain = df.loc[df["Rainfall"].idxmax()]
    max_visits = df.loc[df["Hospital_Visits"].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("### 🌧️ Maximum Rainfall")
            st.info(f"**Month:** {max_rain['Month']}")
            st.info(f"**Amount:** {max_rain['Rainfall']} mm")
    
    with col2:
        with st.container():
            st.markdown("### 🏥 Maximum Hospital Visits")
            st.error(f"**Month:** {max_visits['Month']}")
            st.error(f"**Visits:** {max_visits['Hospital_Visits']}")
    
    st.divider()
    
    # Correlation insights
    with st.expander("🔗 Correlation Analysis with Hospital Visits", expanded=True):
        st.markdown("### Feature Correlations")
        corr_values = df.corr()["Hospital_Visits"].sort_values(ascending=False)
        
        for idx, (feature, value) in enumerate(corr_values.items()):
            if feature != "Hospital_Visits":
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{feature}**")
                with col2:
                    color = "green" if value > 0.5 else "orange" if value > 0 else "red"
                    st.markdown(f":{color}[{value:.3f}]")
                st.progress(abs(value))
                st.markdown("")
    
    st.divider()
    
    # Final summary
    st.success("📌 **Summary:** The dashboard provides comprehensive insights into weather patterns and their correlation with hospital visits!")

# ---- Footer ----
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p>Built with using Streamlit | Data Analytics Dashboard</p>
    </div>
""", unsafe_allow_html=True)

