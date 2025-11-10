import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import numpy as np
from io import BytesIO

# --- Streamlit Setup ---
st.set_page_config(page_title="EDA Analyzer",page_icon="🧭" ,layout="wide")

# --- Custom Dark Theme Styling ---

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e293b", "figure.facecolor": "#0f172a"})

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* ---------- Global UI ---------- */
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
    font-family: 'Poppins', sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #f1f5f9;
    font-weight: 600;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
    font-weight: 600;
    text-align: center;
    margin-bottom: 10px;
}

/* ---------- Buttons ---------- */
div.stButton > button {
    background-color: #2563eb;
    color: #f8fafc;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    transform: translateY(-2px);
}

/* ---------- Download Buttons ---------- */
button[data-baseweb="button"] {
    border-radius: 8px;
    background-color: #22c55e !important;
    color: #ffffff !important;
    border: none;
    transition: all 0.3s ease-in-out;
}

button[data-baseweb="button"]:hover {
    background-color: #16a34a !important;
}

/* ---------- Toggles ---------- */
div[data-testid="stCheckbox"] {
    background-color: #1e293b;
    padding: 8px 12px;
    border-radius: 10px;
    margin-bottom: 8px;
}

div[data-testid="stCheckbox"] label {
    color: #e2e8f0;
    font-weight: 500;
}

/* ---------- Selectboxes ---------- */
div[data-baseweb="select"] {
    background-color: #1e293b;
    color: #e2e8f0;
    border-radius: 10px;
}

/* ---------- Tables ---------- */
.stTable {
    background-color: #1e293b;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(255,255,255,0.05);
}

/* ---------- Dataframes ---------- */
.stDataFrame {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0px 2px 5px rgba(255,255,255,0.05);
}

/* ---------- Alerts (Info/Warning/Success) ---------- */
.stAlert {
    border-radius: 10px;
    font-weight: 500;
    color: #f8fafc;
}

/* Success box */
.stAlert[data-baseweb="notification"] div[role="alert"] {
    background-color: #14532d !important;
}

/* Warning box */
.stAlert[data-baseweb="notification"][class*="warning"] {
    background-color: #78350f !important;
}

/* Info box */
.stAlert[data-baseweb="notification"][class*="info"] {
    background-color: #1e3a8a !important;
}

/* ---------- Charts / Plots ---------- */
/*.stPyplot, .stPlotlyChart, .stImage {
    background-color: #1e293b;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0px 2px 10px rgba(255,255,255,0.05);
    margin-top: 10px;
    margin-bottom: 10px;
}*/

/* ---------- Expanders ---------- */
.streamlit-expander {
    border-radius: 10px !important;
    background-color: #1e293b;
    color: #e2e8f0;
}

.streamlit-expanderHeader {
    font-weight: 600;
    background-color: #334155 !important;
    border-radius: 5px;
}

/* ---------- Sidebar Buttons ---------- */
section[data-testid="stSidebar"] button {
    background-color: #0ea5e9 !important;
    color: #ffffff !important;
    font-weight: 500;
    border: none;
    border-radius: 6px;
}

section[data-testid="stSidebar"] button:hover {
    background-color: #0284c7 !important;
}

/* ---------- Hide Streamlit default footer and menu ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


st.title("EDA Analyzer")


# --- File Upload Section ---
file = st.file_uploader("Upload your CSV or XLSX file", type=["csv", "xlsx"], key="file_uploader")

if file is not None:
    # Read file
    if file.name.endswith(".csv"):
        data = pd.read_csv(file)
        st.subheader("CSV file uploaded successfully!")
    elif file.name.endswith(".xlsx"):
        data = pd.read_excel(file, engine="openpyxl")
        st.subheader("XLSX file uploaded successfully!")
    else:
        st.error("Unsupported file type. Please upload a CSV or XLSX file.")
        st.stop()

    # --- EDA Cleaning Switch ---
    st.subheader("EDA On/Off Switch")
    switch = st.toggle("Switch Power")

    if switch:
        st.success("EDA cleaning is ON")
        df = data.copy()

        # Drop Constant or Unique Columns
        dcu = st.toggle("Drop Constant or Unique Columns")
        if dcu:
            unique_counts = df.nunique()
            to_drop = [col for col, count in unique_counts.items() if count == 1 or count == len(df)]
            df.drop(columns=to_drop, inplace=True)
            st.info(f"Dropped columns: {to_drop}")

        # Handle Missing Values
        hmv = st.toggle("Handle Missing Values")
        if hmv:
            for col, missing_count in df.isnull().sum().items():
                if missing_count > 0:
                    if df[col].dtype != "O":
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            st.info("Missing values handled successfully.")

        # Outlier Treatment
        ot = st.toggle("Outlier Treatment (IQR)")
        if ot:
            num_cols = df.select_dtypes(exclude="O").columns
            for col in num_cols:
                q1 = np.percentile(df[col], 25)
                q3 = np.percentile(df[col], 75)
                iqr = q3 - q1
                lower = q1 - (1.5 * iqr)
                upper = q3 + (1.5 * iqr)
                median = np.median(df[col])
                df.loc[(df[col] < lower) | (df[col] > upper), col] = median
            st.info("Outliers replaced using median (IQR method).")

        # Categorical Encoding
        ctn = st.toggle("Categorical to Numerical Encoding")
        if ctn:
            encoding_method = st.selectbox("Select Encoding Technique", ["Label Encoding", "One-Hot Encoding"], key="encoding_technique")
            if encoding_method == "Label Encoding":
                cat_cols = df.select_dtypes(include=["object"]).columns
                for col in cat_cols:
                    labels = sorted(df[col].astype(str).unique())
                    label_map = {labels[i]: i for i in range(len(labels))}
                    df[col] = df[col].map(label_map)
            elif encoding_method == "One-Hot Encoding":
                df = pd.get_dummies(df, drop_first=True)
            st.info("Categorical encoding applied.")

        # Scaling
        sc = st.toggle("Scale the Data")
        if sc:
            scale_type = st.selectbox("Select Scaling Technique", ["Z Standardization", "Min-Max Scaling"], key="scaling_technique")
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns
            if scale_type == "Z Standardization":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df[num_cols] = scaler.fit_transform(df[num_cols])
            elif scale_type == "Min-Max Scaling":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                df[num_cols] = scaler.fit_transform(df[num_cols])
            st.info(f"Data scaled using {scale_type}.")

        # --- Download Cleaned Data ---
        st.subheader("Download Cleaned Data")
        file_type = st.selectbox("Select the file type to download", ["CSV", "Excel", "JSON"])

        if file_type == "CSV":
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

        elif file_type == "Excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='CleanedData')
            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        elif file_type == "JSON":
            st.download_button(
                label="Download JSON",
                data=df.to_json(orient='records').encode('utf-8'),
                file_name="cleaned_data.json",
                mime="application/json"
            )

    else:
        df = data
        st.warning("EDA cleaning is OFF — Using raw data.")

    # # --- Automated EDA Report ---
    # st.subheader("Automated EDA Report")
    # try:
    #     from ydata_profiling import ProfileReport
    # except ModuleNotFoundError:
    #     st.error("Missing dependency: install it with `pip install ydata-profiling pydantic-settings`")
    #     st.stop()

    # if st.button("Generate Auto EDA Report"):
    #     with st.spinner("Generating report... This might take a minute "):
    #         profile = ProfileReport(df, explorative=True)
    #         report_path = "EDA_Report.html"
    #         profile.to_file(report_path)
    #         st.success("EDA Report generated successfully!")

    #         # Download
    #         with open(report_path, "rb") as f:
    #             st.download_button(
    #                 label="Download EDA Report (HTML)",
    #                 data=f,
    #                 file_name="EDA_Report.html",
    #                 mime="text/html"
    #             )

    #         # Inline Preview
    #         from streamlit.components.v1 import html
    #         with open(report_path, "r", encoding="utf-8") as f:
    #             html_report = f.read()
    #         st.markdown("### EDA Report Preview")
    #         html(html_report, height=800, scrolling=True)

    # --- Data Preview ---
    st.markdown("### Data Preview")
    st.dataframe(df.head())

    # --- Sidebar Controls for Plotting ---
    st.sidebar.header("Plot Options")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Bar Plot", "Pie Chart", "Box Plot", "Histogram", "Scatter Plot",  "Crosstab","Heatmap"], key="plot_type")
    feature1 = None
    feature2 = None
    if plot_type in ["Box Plot", "Histogram", "Pie Chart", "Bar Plot"]:
        feature1 = st.sidebar.selectbox("Select Feature 1 (X-axis)", df.columns.tolist(), key="feature1")

    elif plot_type in ["Scatter Plot", "Crosstab", "Cross Plot"]:
        feature1 = st.sidebar.selectbox("Select Feature 1 (X-axis)", df.columns.tolist(), key="feature1")
        feature2 = st.sidebar.selectbox("Select Feature 2 (Y-axis)", df.columns.tolist(), key="feature2")
    elif plot_type == "Heatmap":
        pass  # No feature selection needed for heatmap

    # --- Session State for Plot Generation ---
    if "generated" not in st.session_state:
        st.session_state.generated = False
        st.session_state.prev_feature1 = None
        st.session_state.prev_feature2 = None
        st.session_state.prev_plot_type = None

    # Detect sidebar changes
    if (feature1 != st.session_state.prev_feature1 or feature2 != st.session_state.prev_feature2 or plot_type != st.session_state.prev_plot_type):
        st.session_state.generated = False

    # Update previous selections
    st.session_state.prev_feature1 = feature1
    st.session_state.prev_feature2 = feature2
    st.session_state.prev_plot_type = plot_type

    # Generate plot button
    if st.sidebar.button("Generate Plot"):
        st.session_state.generated = True

    # --- Plot Section ---
    if st.session_state.generated:
        st.subheader("Plot Result")

        show_freq = st.toggle("Show Frequency Table", key="freq_toggle")

        plt.rcParams["figure.figsize"] = (3.5, 2.5)
        plt.rcParams["figure.dpi"] = 120
        plt.rcParams["axes.titlesize"] = 10

        # Frequency Table
        if plot_type !="Heatmap":
            if show_freq:
                st.markdown("### Frequency Table(s)")
                if feature2 and plot_type in ["Scatter Plot", "Crosstab", "Cross Plot"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### `{feature1}`")
                        f1_table = df[feature1].value_counts().reset_index()
                        f1_table.columns = [feature1, "Frequency"]
                        st.table(f1_table)
                    with col2:
                        st.markdown(f"#### `{feature2}`")
                        f2_table = df[feature2].value_counts().reset_index()
                        f2_table.columns = [feature2, "Frequency"]
                        st.table(f2_table)
                else:
                    freq_table = df[feature1].value_counts().reset_index()
                    freq_table.columns = [feature1, "Frequency"]
                    st.table(freq_table)
        else:
            st.info("Frequency table not applicable for Heatmap.")

        # --- Plot Logic ---
        if plot_type == "Bar Plot":
            fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=120)
            df[feature1].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Bar Plot of {feature1}")
            st.pyplot(fig)

        elif plot_type == "Pie Chart":
            fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=120)
            df[feature1].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart of {feature1}")
            st.pyplot(fig)

        elif plot_type == "Box Plot":
            if pd.api.types.is_numeric_dtype(df[feature1]):
                Q1 = df[feature1].quantile(0.25)
                Q3 = df[feature1].quantile(0.75)
                IQR = Q3 - Q1
                filtered_df = df[(df[feature1] >= Q1 - 1.5 * IQR) & (df[feature1] <= Q3 + 1.5 * IQR)]

                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(3, 2), dpi=120)
                    ax1.boxplot(df[feature1].dropna())
                    ax1.set_title("Before Outlier Handling")
                    st.pyplot(fig1)
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(3, 2), dpi=120)
                    ax2.boxplot(filtered_df[feature1].dropna())
                    ax2.set_title("After Outlier Handling")
                    st.pyplot(fig2)
            else:
                st.warning(f"'{feature1}' is not numeric — Box Plot requires numeric data.")

        elif plot_type == "Histogram":
            if pd.api.types.is_numeric_dtype(df[feature1]):
                fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=120)
                df[feature1].plot(kind="hist", bins=20, ax=ax)
                ax.set_title(f"Histogram of {feature1}")
                st.pyplot(fig)
            else:
                st.warning(f"'{feature1}' is not numeric — Histogram requires numeric data.")

        elif plot_type == "Scatter Plot" and feature2:
            if (pd.api.types.is_numeric_dtype(df[feature1]) and pd.api.types.is_numeric_dtype(df[feature2])):
                fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=120)
                df.plot(kind="scatter", x=feature1, y=feature2, ax=ax)
                ax.set_title(f"Scatter Plot: {feature1} vs {feature2}")
                st.pyplot(fig)
            else:
                st.warning("Both selected columns must be numeric for Scatter Plot.")

        

        elif plot_type == "Crosstab" and feature2:
            st.markdown("### Crosstab Relationship Table")
            ct = pd.crosstab(df[feature1], df[feature2])
            st.dataframe(ct)
            fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
            sns.heatmap(ct, annot=True, cmap="Blues", fmt="d", ax=ax)
            ax.set_title(f"Crosstab Heatmap: {feature1} vs {feature2}")
            st.pyplot(fig)


        elif plot_type == "Heatmap":
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            corr_matrix = df[numeric_cols].corr(numeric_only=True)

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
            plt.title("Correlation Heatmap")
            fig = plt.gcf()
            st.pyplot(fig)


else:
    st.info("Please upload a CSV or XLSX file to start.")
