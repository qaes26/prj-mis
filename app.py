import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
# تأكد أن ملف calculations.py موجود في نفس المجلد
try:
    from calculations import calculate_eoq, calculate_rop, perform_abc_analysis
except ImportError:
    st.error("Missing calculations.py file. Please ensure it is in the same directory.")

# --- Page Config ---
st.set_page_config(
    page_title="Inventory Optimization Tool",
    page_icon="📦",
    layout="wide"
)

# --- Custom CSS ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# --- Helper: Generate Template ---
def get_template_df():
    return pd.DataFrame({
        "Product_Name": ["Widget A", "Widget B", "Widget C"],
        "Annual_Demand": [1000, 500, 200],
        "Ordering_Cost": [50, 50, 50],
        "Holding_Cost": [2, 5, 10],
        "Lead_Time": [5, 3, 7],
        "Unit_Price": [20, 100, 50]
    })

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")
safety_stock_method = st.sidebar.radio("Safety Stock Calculation", ["Fixed Value", "Percentage of Demand"])
global_safety_stock = 0
if safety_stock_method == "Fixed Value":
    global_safety_stock = st.sidebar.number_input("Safety Stock (Units)", min_value=0, value=0)
else:
    ss_pct = st.sidebar.slider("Safety Stock % of Daily Demand", 0.0, 200.0, 50.0)

days_in_year = st.sidebar.number_input("Days in Year", min_value=300, max_value=365, value=365)

# --- Main App ---
st.title("📦 Inventory Optimization Dashboard")
st.markdown("### MIS Project - Education & Analysis Tool")

# Tabs
tab1, tab2, tab3 = st.tabs(["📂 Data Upload", "📊 ABC Analysis", "📉 EOQ & ROP Analysis"])

uploaded_file = None
df = None

with tab1:
    st.header("Upload Inventory Data")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    with col2:
        st.info("Ensure your file has columns: Product_Name, Annual_Demand, Ordering_Cost, Holding_Cost, Lead_Time, Unit_Price")
        st.download_button(
            label="Download Template CSV",
            data=get_template_df().to_csv(index=False),
            file_name="inventory_template.csv",
            mime="text/csv"
        )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # تنظيف أسماء الأعمدة من المسافات الزائدة
            df.columns = df.columns.str.strip()
            
            st.success(f"Successfully loaded {len(df)} products.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")

if df is not None:
    # --- Pre-calculations ---
    required_cols = ["Annual_Demand", "Ordering_Cost", "Holding_Cost", "Lead_Time", "Unit_Price"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    else:
        # تحويل الأعمدة الرقمية لضمان عدم وجود أخطاء في الحسابات
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # الحسابات الإدارية
        df['Daily_Demand'] = df['Annual_Demand'] / days_in_year
        df['EOQ'] = df.apply(lambda x: calculate_eoq(x['Annual_Demand'], x['Ordering_Cost'], x['Holding_Cost']), axis=1)
        
        if safety_stock_method == "Percentage of Demand":
            df['Safety_Stock'] = df['Daily_Demand'] * (ss_pct / 100)
        else:
            df['Safety_Stock'] = global_safety_stock
            
        df['ROP'] = df.apply(lambda x: calculate_rop(x['Daily_Demand'], x['Lead_Time'], x['Safety_Stock']), axis=1)
        df['Annual_Value'] = df['Annual_Demand'] * df['Unit_Price']
        df = perform_abc_analysis(df)
        
        # --- Tab 2: ABC Analysis ---
        with tab2:
            st.subheader("ABC Analysis (Pareto Principle)")
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                fig_abc = px.bar(
                    df, x='Product_Name', y='Annual_Value', color='ABC_Category',
                    title="Pareto Chart: Annual Consumption Value",
                    category_orders={"ABC_Category": ["A", "B", "C"]},
                    color_discrete_map={"A": "#2ecc71", "B": "#f1c40f", "C": "#e74c3c"}
                )
                st.plotly_chart(fig_abc, use_container_width=True)
                
            with col_b:
                abc_counts = df['ABC_Category'].value_counts().sort_index()
                st.write("Item Counts by Class:")
                st.dataframe(abc_counts)
                
                fig_pie = px.pie(
                    df, names='ABC_Category', values='Annual_Value',
                    title="Value Distribution by Class",
                    color='ABC_Category',
                    color_discrete_map={"A": "#2ecc71", "B": "#f1c40f", "C": "#e74c3c"}
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # --- Tab 3: EOQ & ROP Analysis ---
        with tab3:
            st.subheader("Economic Order Quantity (EOQ) & Reorder Point (ROP)")
            
            # تحديد الأعمدة الرقمية فقط للتنسيق العشري لتجنب خطأ السلسلة النصية
            display_cols = ['Product_Name', 'EOQ', 'ROP', 'Safety_Stock', 'Ordering_Cost', 'Holding_Cost']
            num_format_cols = ['EOQ', 'ROP', 'Safety_Stock', 'Ordering_Cost', 'Holding_Cost']
            
            # الحل الجذري لمشكلة التنسيق: استخدام قاموس يحدد التنسيق للأعمدة الرقمية فقط
            st.dataframe(
                df[display_cols].style.format({col: "{:.2f}" for col in num_format_cols})
            )
            
            st.divider()
            st.write("### 🔍 Cost Trade-off Analysis")
            selected_product = st.selectbox("Select Product for Detailed Analysis", df['Product_Name'].unique())
            
            if selected_product:
                prod_data = df[df['Product_Name'] == selected_product].iloc[0]
                order_quantities = np.linspace(1, max(2, prod_data['Annual_Demand']), 100)
                holding_costs = (order_quantities / 2) * prod_data['Holding_Cost']
                ordering_costs = (prod_data['Annual_Demand'] / order_quantities) * prod_data['Ordering_Cost']
                total_costs = holding_costs + ordering_costs
                
                fig_tradeoff = go.Figure()
                fig_tradeoff.add_trace(go.Scatter(x=order_quantities, y=holding_costs, name='Holding Cost'))
                fig_tradeoff.add_trace(go.Scatter(x=order_quantities, y=ordering_costs, name='Ordering Cost'))
                fig_tradeoff.add_trace(go.Scatter(x=order_quantities, y=total_costs, name='Total Cost', line=dict(width=4, color='black')))
                fig_tradeoff.add_vline(x=prod_data['EOQ'], line_dash="dash", annotation_text=f"EOQ: {prod_data['EOQ']:.2f}")
                st.plotly_chart(fig_tradeoff, use_container_width=True)
                
            st.divider()
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Optimization Results')
            st.download_button(
                label="📥 Download Results as Excel",
                data=buffer.getvalue(),
                file_name="inventory_optimization_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Load CSS
local_css("style.css")
