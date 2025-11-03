"""
Manufacturing Command Center - Professional Edition
Clean UI with Working AI Chatbot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import pyodbc
import os
from dotenv import load_dotenv

from ai_models import (
    initialize_ai_models,
    get_ai_equipment_analysis,
    get_ai_root_cause_analysis,
    get_ai_production_insights,
    generate_executive_summary,
    generate_risk_recommendations,
    calculate_roi_impact,
    forecast_production,
    get_openai_client
)

load_dotenv()

st.set_page_config(
    page_title="Manufacturing Command Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header Styling */
    h1 {
        color: #1E40AF;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1E40AF;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #374151;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Card Styling */
    .insight-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #3B82F6;
        margin-bottom: 1.5rem;
    }
    
    .critical-card {
        background: linear-gradient(135deg, #FEF2F2 0%, #FFFFFF 100%);
        border-left-color: #DC2626;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #FFFBEB 0%, #FFFFFF 100%);
        border-left-color: #F59E0B;
    }
    
    .success-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 100%);
        border-left-color: #10B981;
    }
    
    /* Metric Cards */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 3px solid #3B82F6;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E40AF;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .delta-positive {
        color: #10B981;
    }
    
    .delta-negative {
        color: #DC2626;
    }
    
    /* Bullet Points */
    .bullet-list {
        padding-left: 0;
        list-style: none;
    }
    
    .bullet-list li {
        padding: 0.75rem 0;
        border-bottom: 1px solid #F3F4F6;
        position: relative;
        padding-left: 1.5rem;
    }
    
    .bullet-list li:before {
        content: "▸";
        position: absolute;
        left: 0;
        color: #3B82F6;
        font-weight: bold;
    }
    
    .bullet-list li:last-child {
        border-bottom: none;
    }
    
    /* AI Chat Styling */
    .chat-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: #F3F4F6;
        border-left: 3px solid #6B7280;
    }
    
    .ai-message {
        background: #EFF6FF;
        border-left: 3px solid #3B82F6;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Remove extra padding */
    .element-container {
        margin-bottom: 0;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Data frames */
    .dataframe {
        font-size: 0.875rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #F9FAFB;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* ROI Display */
    .roi-display {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #D1FAE5 0%, #FFFFFF 100%);
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    .roi-value {
        font-size: 3rem;
        font-weight: 700;
        color: #10B981;
        margin: 0;
    }
    
    .roi-label {
        font-size: 1rem;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    /* Enhanced Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #F9FAFB;
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 65px;
        background-color: white;
        border-radius: 10px;
        padding: 0 32px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #6B7280;
        border: 2px solid transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #F3F4F6;
        color: #374151;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white !important;
        border: 2px solid #1E40AF;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        font-size: 1.15rem;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

ist = pytz.timezone('Asia/Kolkata')

# ==================== CONNECTION ====================
FABRIC_SERVER = os.getenv('FABRIC_SERVER')
FABRIC_DATABASE = os.getenv('FABRIC_DATABASE')

def get_fabric_connection():
    """Create a new Fabric connection - do not cache to avoid stale connections"""
    try:
        connection_string = (
            f"Driver={{ODBC Driver 18 for SQL Server}};"
            f"Server=tcp:{FABRIC_SERVER},1433;"
            f"Database={FABRIC_DATABASE};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
            f"Authentication=ActiveDirectoryInteractive;"
        )
        conn = pyodbc.connect(connection_string)
        return conn
    except Exception as e:
        st.error(f"❌ Connection Error: {str(e)}")
        st.info("💡 Troubleshooting tips:\n- Run `az login` to authenticate\n- Verify FABRIC_SERVER and FABRIC_DATABASE in .env\n- Check your Fabric workspace access")
        return None

@st.cache_data(ttl=30)
def load_data_from_fabric():
    """Load data from Fabric with proper connection management"""
    conn = None
    try:
        conn = get_fabric_connection()
        if conn is None:
            return None, None, None, None, None
        
        # Load all data
        df_equipment_summary = pd.read_sql("SELECT * FROM silver_equipment_summary", conn)
        df_orders = pd.read_sql("SELECT * FROM silver_fact_production_orders", conn)
        df_sensors = pd.read_sql("SELECT TOP 1000 * FROM silver_fact_sensor_readings ORDER BY timestamp DESC", conn)
        df_line_summary = pd.read_sql("SELECT * FROM silver_production_line_summary", conn)
        df_equipment = pd.read_sql("SELECT * FROM silver_dim_equipment", conn)
        
        return df_equipment_summary, df_orders, df_sensors, df_line_summary, df_equipment
        
    except Exception as e:
        st.error(f"❌ Database Error: {str(e)}")
        st.info("💡 Please verify that the tables exist in your Fabric database:\n- silver_equipment_summary\n- silver_fact_production_orders\n- silver_fact_sensor_readings\n- silver_production_line_summary\n- silver_dim_equipment")
        return None, None, None, None, None
    finally:
        # Always close the connection
        if conn is not None:
            try:
                conn.close()
            except:
                pass

# ==================== LOAD DATA ====================
with st.spinner("🔄 Loading data and initializing AI..."):
    data = load_data_from_fabric()
    
    if all(d is not None for d in data):
        df_equipment_summary, df_orders, df_sensors, df_line_summary, df_equipment = data
        ai_models = initialize_ai_models(df_sensors)
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
    else:
        st.error("❌ Failed to load data")
        st.stop()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("# 🏭 Command Center")
    st.markdown("---")
    
    st.markdown("### 📡 System Status")
    st.success("🟢 Connected")
    st.success("🟢 AI Active")
    st.info(f"⏰ {datetime.now(ist).strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    st.markdown("### 📊 Quick Stats")
    st.metric("Equipment", len(df_equipment), border=True)
    st.metric("Orders", len(df_orders), border=True)
    st.metric("Alerts", int(df_equipment_summary['maintenance_alert_count'].sum()), border=True)
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("v2.5.0 AI | Powered by OpenAI")

# ==================== MAIN CONTENT WITH TABS ====================

# Add title above tabs
st.markdown("""
<div style="text-align: center; margin-bottom: 1.5rem;">
    <h1 style="color: #1E40AF; font-size: 2.5rem; font-weight: 700; margin: 0;">
        🏭 Manufacturing Command Center
    </h1>
    <p style="color: #6B7280; font-size: 1rem; margin-top: 0.5rem;">
        Real-time AI operational intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# Create tabs at the top of the main page
tab1, tab2, tab3, tab4 = st.tabs([
    "🌅 Morning Briefing",
    "📦 Order Fulfillment", 
    "⚡ Predictive Maintenance",
    "🤖 AI Assistant"
])

# ==================== TAB 1: MORNING BRIEFING ====================
with tab1:
    st.markdown("# 🌅 Morning Briefing")
    st.markdown(f"**{datetime.now(ist).strftime('%A, %B %d, %Y')}** • Real-time AI operational intelligence")
    
    st.markdown("")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_oee = df_equipment_summary['avg_oee'].mean()
        st.metric(
            label="🎯 Average OEE",
            value=f"{avg_oee:.1f}%",
            delta="↑ 2.5%",
            delta_color="normal"
        )
    
    with col2:
        total_prod = df_equipment_summary['total_units_produced'].sum()
        st.metric(
            label="📦 Production",
            value=f"{int(total_prod):,}",
            delta="↑ 127 units",
            delta_color="normal"
        )
    
    with col3:
        on_track = len(df_orders[df_orders['status'] == 'On Track'])
        st.metric(
            label="✅ On Track",
            value=f"{on_track}/{len(df_orders)}",
            delta=f"{(on_track/len(df_orders)*100):.0f}%",
            delta_color="normal"
        )
    
    with col4:
        alerts = int(df_equipment_summary['maintenance_alert_count'].sum())
        st.metric(
            label="🚨 Alerts",
            value=str(alerts),
            delta="Requires Action",
            delta_color="inverse"
        )
    
    st.markdown("")
    
    # AI Executive Summary
    st.markdown("## 🤖 AI Executive Summary")
    
    with st.spinner("🤖 Analyzing operations..."):
        exec_summary = generate_executive_summary(df_equipment_summary, df_orders, df_line_summary)
    
    st.info(exec_summary)
    
    # AI Risk Recommendations
    st.markdown("## 🎯 Priority Actions")
    
    with st.spinner("🤖 Generating recommendations..."):
        recommendations = generate_risk_recommendations(df_equipment_summary, df_orders, df_line_summary)
    
    if recommendations:
        for idx, rec in enumerate(recommendations):
            if rec['priority'] in ['CRITICAL', 'Critical']:
                container_type = "error"
                icon = "🔴"
            elif rec['priority'] in ['HIGH', 'High']:
                container_type = "warning"
                icon = "🟡"
            else:
                container_type = "success"
                icon = "🟢"
            
            # Use appropriate Streamlit container
            if container_type == "error":
                with st.container():
                    st.error(f"### {icon} {rec['title']}")
                    st.markdown("**Key Points:**")
                    st.markdown(f"- **Priority:** {rec['priority']}")
                    st.markdown(f"- **Type:** {rec['type']}")
                    st.markdown(f"- **Impact if Ignored:** {rec['impact']}")
                    st.markdown(f"- **Recommended Action:** {rec['action']}")
                    if rec['savings'] > 0:
                        st.markdown(f"- **Estimated Savings:** ${rec['savings']:,}")
            elif container_type == "warning":
                with st.container():
                    st.warning(f"### {icon} {rec['title']}")
                    st.markdown("**Key Points:**")
                    st.markdown(f"- **Priority:** {rec['priority']}")
                    st.markdown(f"- **Type:** {rec['type']}")
                    st.markdown(f"- **Impact if Ignored:** {rec['impact']}")
                    st.markdown(f"- **Recommended Action:** {rec['action']}")
                    if rec['savings'] > 0:
                        st.markdown(f"- **Estimated Savings:** ${rec['savings']:,}")
            else:
                with st.container():
                    st.success(f"### {icon} {rec['title']}")
                    st.markdown("**Key Points:**")
                    st.markdown(f"- **Priority:** {rec['priority']}")
                    st.markdown(f"- **Type:** {rec['type']}")
                    st.markdown(f"- **Impact if Ignored:** {rec['impact']}")
                    st.markdown(f"- **Recommended Action:** {rec['action']}")
                    if rec['savings'] > 0:
                        st.markdown(f"- **Estimated Savings:** ${rec['savings']:,}")
            
            st.markdown("")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("✅ Approve Action", key=f"approve_{idx}", use_container_width=True):
                    st.success("✅ Action approved and scheduled")
            with col_b:
                if st.button("📋 Create Work Order", key=f"wo_{idx}", use_container_width=True):
                    st.success("✅ Work order created")
            with col_c:
                if st.button("📊 View Details", key=f"details_{idx}", use_container_width=True):
                    st.info("Detailed analysis available in equipment section")
    
    
    # Equipment Health Issues
    critical_equipment = df_equipment_summary[
        (df_equipment_summary['maintenance_alert_count'] > 0) | 
        (df_equipment_summary['avg_health_score'] < 80)
    ]
    
    if len(critical_equipment) > 0:
        st.markdown("## ⚙️ Equipment Requiring Attention")
        
        for _, eq in critical_equipment.iterrows():
            with st.expander(f"⚠️ {eq['equipment_id']} - {eq['equipment_name']} (Health: {eq['avg_health_score']:.0f}%)", expanded=False):
                
                equipment_sensors = df_sensors[df_sensors['equipment_id'] == eq['equipment_id']]
                
                if len(equipment_sensors) > 0:
                    latest_sensor = equipment_sensors.iloc[0]
                    
                    # ML Prediction
                    prediction = ai_models['maintenance'].predict_failure(latest_sensor)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Failure Risk", prediction['risk_level'])
                    with col2:
                        st.metric("Time to Failure", prediction['time_to_failure'])
                    with col3:
                        st.metric("Probability", f"{prediction['failure_probability']*100:.0f}%")
                    
                    st.warning(f"**AI Recommendation:** {prediction['recommendation']}")
                    
                    st.markdown("")
                    
                    # Detailed ROI Analysis with Cost Justifications
                    st.markdown("### 💰 ROI Analysis: AI-Driven Preventive Maintenance")
                    
                    with st.spinner("Calculating ROI based on equipment data..."):
                        roi = calculate_roi_impact('maintenance', eq, latest_sensor)
                    
                    # Summary Comparison
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.error("**❌ Without AI Action (Reactive)**")
                        st.metric("Total Cost", f"${roi['prevented_cost']:,}", delta="Equipment failure scenario", delta_color="inverse")
                    with col_y:
                        st.success("**✅ With AI Action (Proactive)**")
                        st.metric("Total Cost", f"${roi['action_cost']:,}", delta="Preventive maintenance", delta_color="normal")
                    
                    # Detailed Cost Breakdown - Without AI (expanded by default)
                    st.markdown("---")
                    with st.expander("🔴 **Cost Breakdown: WITHOUT AI Action** (Equipment Failure Scenario)", expanded=True):
                        st.markdown("#### What Happens If We DON'T Act:")
                        
                        if 'prevented_breakdown' in roi and roi['prevented_breakdown']:
                            total_check = 0
                            for cost_item, details in roi['prevented_breakdown'].items():
                                st.markdown(f"**{cost_item.replace('_', ' ').title()}:** ${details['cost']:,}")
                                st.caption(f"📊 Calculation: {details['reason']}")
                                st.info(f"💡 Why: {details['detail']}")
                                st.markdown("")
                                total_check += details['cost']
                            
                            st.markdown("---")
                            st.error(f"**Total Cost Without AI Action: ${total_check:,}**")
                            st.caption(f"Based on equipment health at {roi.get('health_score', 0):.0f}%")
                    
                    # Detailed Cost Breakdown - With AI (expanded by default)
                    with st.expander("🟢 **Cost Breakdown: WITH AI Action** (Preventive Maintenance)", expanded=True):
                        st.markdown("#### What We Invest When We Act Proactively:")
                        
                        if 'action_breakdown' in roi and roi['action_breakdown']:
                            total_check = 0
                            for cost_item, details in roi['action_breakdown'].items():
                                st.markdown(f"**{cost_item.replace('_', ' ').title()}:** ${details['cost']:,}")
                                st.caption(f"📊 Calculation: {details['reason']}")
                                st.success(f"✅ Benefit: {details['detail']}")
                                st.markdown("")
                                total_check += details['cost']
                            
                            st.markdown("---")
                            st.success(f"**Total Cost With AI Action: ${total_check:,}**")
                            st.caption("Planned maintenance scheduled during optimal time")
                    
                    # Additional Benefits
                    if 'additional_benefits' in roi and roi['additional_benefits']:
                        st.markdown("---")
                        st.markdown("#### 🎁 Additional Benefits (Not Included in Cost Calculation)")
                        for benefit in roi['additional_benefits']:
                            st.info(f"**{benefit['benefit']}:** {benefit['value']}")
                    
                    # Clear Business Case
                    st.markdown("---")
                    st.markdown("#### 📊 Business Case Summary")
                    
                    if roi['savings'] > 0:
                        savings_ratio = roi['savings'] / roi['action_cost'] if roi['action_cost'] > 0 else 0
                        st.success(f"""
                        **✅ RECOMMENDATION: Proceed with preventive maintenance**
                        
                        - **Investment Required:** ${roi['action_cost']:,}
                        - **Cost Prevented:** ${roi['prevented_cost']:,}
                        - **Net Savings:** ${roi['savings']:,}
                        - **Return Multiple:** {savings_ratio:.1f}x investment
                        
                        **Payback:** Immediate (avoids much larger future cost)
                        """)
                    
                    st.markdown("---")
                    
                    # Action Buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ Schedule Maintenance", key=f"sched_{eq['equipment_id']}", use_container_width=True):
                            st.success(f"✅ Maintenance scheduled for {eq['equipment_id']}")
                    with col_b:
                        if st.button("🤖 Get AI Analysis", key=f"ai_{eq['equipment_id']}", use_container_width=True):
                            with st.spinner("🤖 Analyzing..."):
                                analysis = get_ai_equipment_analysis(eq, latest_sensor)
                                st.info(analysis)

# ==================== TAB 2: ORDER FULFILLMENT ====================
with tab2:
    st.markdown("# 📦 Order Fulfillment Tracker")
    st.markdown("AI-assisted production tracking and optimization")
    
    st.markdown("")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_ordered = df_orders['quantity_ordered'].sum()
    total_produced = df_orders['quantity_produced'].sum()
    on_track = len(df_orders[df_orders['status'] == 'On Track'])
    at_risk = len(df_orders[df_orders['risk_score'].isin(['High', 'Medium'])])
    
    with col1:
        st.metric(
            label="Progress",
            value=f"{(total_produced/total_ordered*100):.1f}%",
            delta=f"{int(total_produced):,} / {int(total_ordered):,}"
        )
    
    with col2:
        st.metric(
            label="On Track",
            value=str(on_track),
            delta=f"{(on_track/len(df_orders)*100):.0f}%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="At Risk",
            value=str(at_risk),
            delta="Requires Action",
            delta_color="inverse"
        )
    
    with col4:
        avg_comp = df_orders['completion_percent'].mean()
        st.metric(
            label="Avg Completion",
            value=f"{avg_comp:.1f}%"
        )
    
    st.markdown("")
    
    # AI Production Insights
    st.markdown("## 🤖 AI Production Intelligence")
    
    with st.spinner("🤖 Analyzing production..."):
        insights = get_ai_production_insights(df_orders, df_line_summary)
    
    st.info(insights)
    
    # Production Lines with Intelligent Risk Analysis
    st.markdown("## 🏭 Production Lines - Intelligent Risk Analysis")
    
    for line in sorted(df_line_summary['production_line'].unique()):
        line_data = df_line_summary[df_line_summary['production_line'] == line].iloc[0]
        line_orders = df_orders[df_orders['production_line'] == line].copy()
        avg_completion = line_orders['completion_percent'].mean()
        
        # CALCULATE REAL RISK - Ignore database risk_score column
        def calculate_real_risk(row):
            """Calculate actual risk based on completion and timeline"""
            completion = row['completion_percent']
            days_left = row['days_until_due']
            
            # Calculate what needs to be done vs time available
            completion_gap = 100 - completion
            
            # Critical Risk Scenarios
            if completion < 50 and days_left <= 2:
                return 'CRITICAL', 95  # Less than half done, 2 days or less
            elif completion < 70 and days_left <= 3:
                return 'CRITICAL', 90  # Behind schedule with little time
            
            # High Risk Scenarios  
            elif completion < 60 and days_left <= 5:
                return 'HIGH', 75  # Significantly behind
            elif completion < 80 and days_left <= 2:
                return 'HIGH', 70  # Tight timeline
            
            # Medium Risk Scenarios
            elif completion < 85 and days_left <= 3:
                return 'MEDIUM', 50  # Moderate concern
            elif completion < 90 and days_left <= 2:
                return 'MEDIUM', 45  # Needs monitoring
            
            # Check if behind expected pace
            elif days_left > 0:
                # Assume linear progress - should be at (total_days - days_left) / total_days * 100%
                # Estimate total days (rough approximation)
                if completion < 95 and days_left == 1:
                    return 'MEDIUM', 40  # Last day, not complete
                
            # Low or No Risk
            if completion >= 95:
                return 'NONE', 0  # Essentially complete
            elif completion >= 100:
                return 'NONE', 0  # Complete
            else:
                return 'LOW', 20  # On track
        
        # Apply risk calculation to all orders
        line_orders[['calculated_risk', 'risk_score_num']] = line_orders.apply(
            lambda row: pd.Series(calculate_real_risk(row)), axis=1
        )
        
        # Filter to actually at-risk orders (not the fake database risk_score)
        at_risk_orders = line_orders[line_orders['calculated_risk'].isin(['CRITICAL', 'HIGH', 'MEDIUM'])].copy()
        critical_orders = line_orders[line_orders['calculated_risk'] == 'CRITICAL']
        
        # Sort by risk score (highest first)
        at_risk_orders = at_risk_orders.sort_values('risk_score_num', ascending=False)
        
        if avg_completion >= 90:
            icon = "🟢"
        elif avg_completion >= 75:
            icon = "🟡"
        else:
            icon = "🔴"
        
        with st.expander(f"{icon} **{line}** - {len(line_orders)} Orders ({len(at_risk_orders)} actually at risk)", expanded=len(at_risk_orders) > 0):
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("OEE", f"{line_data['line_oee']:.1f}%")
            with col2:
                st.metric("Utilization", f"{line_data['line_utilization']:.1f}%")
            with col3:
                st.metric("At Risk Orders", len(at_risk_orders), delta="Requires Action" if len(at_risk_orders) > 0 else "All On Track", delta_color="inverse" if len(at_risk_orders) > 0 else "normal")
            with col4:
                st.metric("Avg Completion", f"{avg_completion:.0f}%")
            
            # Risk Analysis - Show ONLY truly at-risk orders
            if len(at_risk_orders) > 0:
                st.markdown("---")
                st.markdown("### ⚠️ Orders Requiring Attention")
                
                for _, order in at_risk_orders.iterrows():
                    # Determine risk factors based on REAL analysis
                    risk_factors = []
                    recommendations = []
                    
                    completion = order['completion_percent']
                    days_left = order['days_until_due']
                    completion_gap = 100 - completion
                    daily_required_rate = completion_gap / max(days_left, 0.5)
                    
                    # Identify specific risks
                    if order['calculated_risk'] == 'CRITICAL':
                        if completion < 50 and days_left <= 2:
                            risk_factors.append(f"🚨 **CRITICAL: Severe Timeline Risk** - Only {completion:.0f}% complete with {days_left} day(s) remaining")
                            risk_factors.append(f"📊 **Completion Gap**: {completion_gap:.0f}% still needed")
                            recommendations.append("🔴 **URGENT**: Stop other work and focus all resources on this order")
                            recommendations.append("📞 **Escalate immediately** to plant manager")
                            recommendations.append("📱 **Contact customer** about potential delay")
                        elif completion < 70 and days_left <= 3:
                            risk_factors.append(f"⚠️ **CRITICAL: Behind Schedule** - {completion:.0f}% complete, {days_left} days left")
                            risk_factors.append(f"⏱️ **Required Rate**: {daily_required_rate:.1f}% per day (very aggressive)")
                            recommendations.append("🔴 **Priority Action**: Assign additional shifts/overtime")
                            recommendations.append("📋 **Daily monitoring** required")
                    
                    elif order['calculated_risk'] == 'HIGH':
                        if completion < 60 and days_left <= 5:
                            risk_factors.append(f"⏰ **High Risk: Low Completion** - {completion:.0f}% done, {days_left} days to deadline")
                            risk_factors.append(f"📈 **Daily Target**: Need {daily_required_rate:.1f}% completion per day")
                            recommendations.append("🟡 **Action Required**: Add extra resources or shift")
                            recommendations.append("📊 **Monitor closely**: Track daily progress")
                        elif completion < 80 and days_left <= 2:
                            risk_factors.append(f"⏰ **High Risk: Tight Timeline** - {completion:.0f}% complete, only {days_left} day(s) left")
                            risk_factors.append(f"⚡ **Acceleration Needed**: {daily_required_rate:.1f}% per day required")
                            recommendations.append("🟡 **Expedite**: Prioritize this order over others")
                            recommendations.append("⚙️ **Check capacity**: Ensure line can handle required rate")
                    
                    elif order['calculated_risk'] == 'MEDIUM':
                        if completion < 85 and days_left <= 3:
                            risk_factors.append(f"📊 **Moderate Risk: Needs Monitoring** - {completion:.0f}% complete, {days_left} days remaining")
                            risk_factors.append(f"📈 **Target Rate**: {daily_required_rate:.1f}% per day to finish on time")
                            recommendations.append("👁️ **Monitor**: Check progress daily")
                            recommendations.append("🎯 **Stay on track**: Maintain current pace or better")
                        elif completion < 90 and days_left <= 2:
                            risk_factors.append(f"⏱️ **Moderate Risk: Close Deadline** - {completion:.0f}% done, {days_left} day(s) left")
                            recommendations.append("👁️ **Watch closely**: Small delays could cause issues")
                            recommendations.append("📋 **Plan buffer**: Prepare contingency if needed")
                        elif completion < 95 and days_left == 1:
                            risk_factors.append(f"⏰ **Last Day Risk** - {completion:.0f}% complete, due tomorrow")
                            risk_factors.append(f"🎯 **Remaining**: {completion_gap:.0f}% must be completed today")
                            recommendations.append("🔍 **Focus**: Ensure completion today")
                            recommendations.append("⚡ **No delays**: Any issues will cause late delivery")
                    
                    # Check OEE impact on ALL at-risk orders
                    if line_data['line_oee'] < 75:
                        risk_factors.append(f"⚙️ **Line Efficiency Issue**: OEE at {line_data['line_oee']:.1f}% reduces production capacity")
                        recommendations.append("🔧 **Maintenance needed**: Improve line OEE to increase throughput")
                    
                    # Only show if there are actual risks
                    if len(risk_factors) > 0:
                        # Display order card with appropriate severity
                        if order['calculated_risk'] == 'CRITICAL':
                            st.error(f"🔴 **CRITICAL: Order {order['order_id']}** - {order['customer']}")
                        elif order['calculated_risk'] == 'HIGH':
                            st.warning(f"🟡 **HIGH RISK: Order {order['order_id']}** - {order['customer']}")
                        else:
                            st.info(f"🔵 **MONITOR: Order {order['order_id']}** - {order['customer']}")
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Completion", f"{completion:.0f}%")
                        with col_b:
                            st.metric("Days to Due", days_left)
                        with col_c:
                            st.metric("Quantity", f"{int(order['quantity_produced'])}/{int(order['quantity_ordered'])}")
                        with col_d:
                            st.metric("Risk Level", order['calculated_risk'])
                        
                        # Show risk factors
                        st.markdown("**🔍 Why This Order is At Risk:**")
                        for factor in risk_factors:
                            st.markdown(f"• {factor}")
                        
                        # Show recommendations
                        if recommendations:
                            st.markdown("**💡 What To Do:**")
                            for i, rec in enumerate(recommendations, 1):
                                st.markdown(f"{i}. {rec}")
                        
                        # Calculate projection
                        if days_left > 0 and line_data['total_production'] > 0:
                            remaining_qty = order['quantity_ordered'] - order['quantity_produced']
                            daily_rate = line_data['total_production'] / len(line_orders)
                            estimated_days = remaining_qty / max(daily_rate, 1)
                            
                            if estimated_days > days_left:
                                days_late = int(estimated_days - days_left)
                                st.error(f"⏰ **Projection**: Will finish **{days_late} day(s) LATE** at current production rate")
                                st.markdown(f"**Current pace**: {daily_rate:.0f} units/day → **Need**: {remaining_qty / days_left:.0f} units/day")
                            else:
                                days_early = int(days_left - estimated_days)
                                if days_early > 1:
                                    st.success(f"✅ **Projection**: Will finish {days_early} days early if pace maintained")
                                else:
                                    st.info(f"⏱️ **Projection**: On track to finish just in time")
                        
                        st.markdown("---")
                
                # Overall line recommendations
                if len(critical_orders) > 0 or len(at_risk_orders) > 2:
                    st.markdown("### 📊 Production Line Overall Actions")
                    
                    if len(critical_orders) > 0:
                        st.error(f"🚨 **URGENT**: {len(critical_orders)} order(s) in CRITICAL status - immediate management intervention required")
                    
                    if len(at_risk_orders) > 2:
                        st.warning(f"⚠️ **High Risk Count**: {len(at_risk_orders)} orders need attention - review line capacity and scheduling")
                    
                    if line_data['line_oee'] < 75:
                        st.info(f"⚙️ **Root Cause**: Line OEE at {line_data['line_oee']:.1f}% - improving this will help ALL orders")
                        
            else:
                st.success("✅ **All Orders On Track** - No immediate risks identified")
                st.markdown(f"**Line Performance:** {avg_completion:.1f}% average completion, {line_data['line_oee']:.1f}% OEE")
                
                # Show well-performing orders
                on_track = line_orders[line_orders['completion_percent'] >= 90]
                if len(on_track) > 0:
                    st.info(f"🎯 **{len(on_track)} order(s) are 90%+ complete** - excellent progress!")

    st.markdown("## 📋 Order Details")
    
    display_orders = df_orders[[
        'order_id', 'customer', 'production_line', 
        'completion_percent', 'days_until_due', 'risk_score', 'status'
    ]].copy()
    
    st.dataframe(
        display_orders.style.background_gradient(subset=['completion_percent'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True
    )

# ==================== TAB 3: PREDICTIVE MAINTENANCE ====================
with tab3:
    st.markdown("# ⚡ Predictive Maintenance")
    st.markdown("ML-powered failure predictions and anomaly detection")
    
    st.markdown("")
    
    # AI Predictions
    st.markdown("## 🤖 AI Failure Predictions")
    
    predictions_list = []
    
    for _, eq in df_equipment_summary.iterrows():
        equipment_sensors = df_sensors[df_sensors['equipment_id'] == eq['equipment_id']]
        
        if len(equipment_sensors) > 0:
            latest_sensor = equipment_sensors.iloc[0]
            pred = ai_models['maintenance'].predict_failure(latest_sensor)
            
            predictions_list.append({
                'Equipment': eq['equipment_id'],
                'Name': eq['equipment_name'],
                'Health': f"{eq['avg_health_score']:.0f}%",
                'Risk': pred['risk_level'],
                'Probability': f"{pred['failure_probability']*100:.0f}%",
                'Time to Failure': pred['time_to_failure'],
                'Recommendation': pred['recommendation']
            })
    
    predictions_df = pd.DataFrame(predictions_list)
    st.dataframe(predictions_df, use_container_width=True, hide_index=True)
    
    # Anomaly Detection
    st.markdown("## 🔍 Real-Time Anomaly Detection")
    
    anomalies_found = False
    
    for _, eq in df_equipment_summary.iterrows():
        equipment_sensors = df_sensors[df_sensors['equipment_id'] == eq['equipment_id']]
        
        if len(equipment_sensors) > 0:
            latest_sensor = equipment_sensors.iloc[0]
            anomaly_result = ai_models['anomaly'].detect_anomalies(latest_sensor)
            
            if anomaly_result['is_anomaly']:
                anomalies_found = True
                
                with st.container():
                    st.error(f"### 🚨 {eq['equipment_id']} - Anomaly Detected")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Severity", anomaly_result['severity'])
                    with col2:
                        st.metric("Anomaly Score", f"{anomaly_result['anomaly_score']:.2f}")
                    with col3:
                        st.metric("Status", "🔴 Anomalous")
                    
                    st.markdown(f"**Explanation:** {anomaly_result['explanation']}")
                    
                    if st.button(f"🔬 Get Root Cause Analysis", key=f"rca_{eq['equipment_id']}", use_container_width=True):
                        with st.spinner("🤖 Analyzing root cause..."):
                            issues = anomaly_result['explanation'].split(';')
                            rca = get_ai_root_cause_analysis(eq['equipment_id'], issues)
                            st.info(rca)
    
    if not anomalies_found:
        st.success("✅ No anomalies detected - all equipment operating normally")
    
    # Production Forecasting
    st.markdown("## 📈 AI Production Forecast")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_eq = st.selectbox("Select Equipment", sorted(df_sensors['equipment_id'].unique()))
    with col2:
        days = st.slider("Days Ahead", 1, 14, 7)
    with col3:
        st.markdown("")
        forecast_btn = st.button("🔮 Generate Forecast", use_container_width=True)
    
    if forecast_btn:
        with st.spinner("🤖 Forecasting production..."):
            forecast_df = forecast_production(df_sensors, selected_eq, days)
            
            fig = px.line(
                forecast_df, 
                x='date', 
                y='forecasted_units',
                title=f"Production Forecast - {selected_eq}",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Forecasted Units",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            total_forecast = forecast_df['forecasted_units'].sum()
            st.success(f"**📊 Forecast Summary:** Expected production of **{int(total_forecast):,} units** over next {days} days")

# ==================== TAB 4: AI ASSISTANT ====================
with tab4:
    st.markdown("# 🤖 AI Assistant")
    st.markdown("Ask questions and get instant AI-powered insights about your operations")
    
    st.markdown("")
    
    # Quick Action Buttons
    st.markdown("### 💡 Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 Equipment Health Issues?", use_container_width=True):
            st.session_state.quick_query = "What equipment health issues should I prioritize today?"
    
    with col2:
        if st.button("📊 Production Summary?", use_container_width=True):
            st.session_state.quick_query = "Give me a production summary for today"
    
    with col3:
        if st.button("💰 ROI Performance?", use_container_width=True):
            st.session_state.quick_query = "What's the ROI of our AI system this month?"
    
    st.markdown("")
    
    # Chat Interface
    st.markdown("### 💬 Chat with AI")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message ai-message">🤖 <strong>AI Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your question here...")
    
    # Handle quick query
    if 'quick_query' in st.session_state:
        user_input = st.session_state.quick_query
        del st.session_state.quick_query
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate AI response
        with st.spinner("🤖 AI thinking..."):
            client = get_openai_client()
            
            if client:
                # Build context from data
                context = f"""
                You are an AI assistant for a manufacturing command center. Answer questions based on this data:
                
                Equipment Status:
                - Total Equipment: {len(df_equipment_summary)}
                - Critical Issues: {len(df_equipment_summary[df_equipment_summary['avg_health_score'] < 80])}
                - Average OEE: {df_equipment_summary['avg_oee'].mean():.1f}%
                - Maintenance Alerts: {int(df_equipment_summary['maintenance_alert_count'].sum())}
                
                Production Orders:
                - Total Orders: {len(df_orders)}
                - On Track: {len(df_orders[df_orders['status'] == 'On Track'])}
                - At Risk: {len(df_orders[df_orders['risk_score'].isin(['High', 'Medium'])])}
                - Avg Completion: {df_orders['completion_percent'].mean():.1f}%
                
                Production Lines:
                {df_line_summary[['production_line', 'line_oee', 'active_orders']].to_string(index=False)}
                
                Provide helpful, specific answers in 3-4 sentences. Use bullet points if listing multiple items.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": context},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0.3,
                        max_tokens=300
                    )
                    
                    ai_response = response.choices[0].message.content
                    
                except Exception as e:
                    ai_response = f"I'm having trouble connecting right now. Based on your data: You have {len(df_equipment_summary[df_equipment_summary['avg_health_score'] < 80])} equipment requiring attention, {len(df_orders[df_orders['status'] == 'On Track'])}/{len(df_orders)} orders on track, and an average OEE of {df_equipment_summary['avg_oee'].mean():.1f}%."
            else:
                ai_response = "AI service unavailable. Please check your OpenAI API key configuration."
        
        # Add AI response
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Rerun to show new messages
        st.rerun()

# Footer
st.markdown("---")
st.caption(f"🏭 Manufacturing Command Center v2.5.0 AI | {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')} | Powered by Systech")