"""
Agentic Data Cleaner - Streamlit Interface
Interactive data cleaning with AI reasoning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from agentic_data_cleaner import AgenticDataCleaner, CleaningStep
import io

# Page configuration
st.set_page_config(
    page_title="🤖 Aka Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .reasoning-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .step-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    /* NEW: Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #4ECDC4;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload' 
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'cleaning_plan' not in st.session_state:
    st.session_state.cleaning_plan = []
if 'selected_steps' not in st.session_state:
    st.session_state.selected_steps = []
# NEW: For wizard and explain mode
if 'wizard_step' not in st.session_state:
    st.session_state.wizard_step = 0  
if 'explain_mode' not in st.session_state:
    st.session_state.explain_mode = False
if 'step_overrides' not in st.session_state:
    st.session_state.step_overrides = {}  

# Header
st.markdown('<h1 class="main-header">🤖 Aka Cleaner</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Lets clean together</p>', unsafe_allow_html=True)

# Sidebar - Progress tracker + NEW: Explain mode toggle
with st.sidebar:
    st.header("📊 Progress Tracker")
    
    stages = {
        'upload': '📁 Upload Data',
        'target': '🎯 Target Selection',
        'sanity': '✅ Sanity Check',
        'eda': '📈 EDA',
        'cleaning': '🧹 Cleaning',
        'done': '✨ Complete'
    }
    
    for stage_key, stage_name in stages.items():
        if st.session_state.stage == stage_key:
            st.markdown(f"**➡️ {stage_name}**")
        elif list(stages.keys()).index(stage_key) < list(stages.keys()).index(st.session_state.stage):
            st.markdown(f"✅ ~~{stage_name}~~")
        else:
            st.markdown(f"⏸️ {stage_name}")
    
    st.divider()
    
    if st.session_state.df is not None:
        st.header("📋 Dataset Info")
        st.metric("Rows", int(st.session_state.df.shape[0]))
        st.metric("Columns", int(st.session_state.df.shape[1]))
        
        if st.session_state.target_column:
            st.metric("Target", st.session_state.target_column)
    
    st.divider()
    
    # Explain mode toggle
    st.session_state.explain_mode = st.toggle("🧠 Explain Mode", value=st.session_state.explain_mode, help="Turn on for detailed explanations and learning tips!")
    
    if st.button("🔄 Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Onboarding Wizard Logic
if st.session_state.wizard_step == 0 and st.session_state.stage == 'upload':
    # Detect first-time or beginner (for now, always show on upload)
    st.session_state.wizard_step = 1

wizard_steps = [
    {"title": "Welcome to Aka Cleaner!", "content": "With this tool, We try cleaning data together. Let's start by uploading your file.", "button": "Got it!"},
    {"title": "Upload Your Data", "content": "Choose a CSV, Excel, or JSON file. We'll preview it for you.", "button": "Next"},

]

if 1 <= st.session_state.wizard_step <= len(wizard_steps):
    with st.expander(f"Let's do it🤗: Step {st.session_state.wizard_step}", expanded=True):
        step = wizard_steps[st.session_state.wizard_step - 1]
        st.markdown(f"### {step['title']}")
        st.write(step['content'])
        if st.button(step['button']):
            if st.session_state.wizard_step < len(wizard_steps):
                st.session_state.wizard_step += 1
            else:
                st.session_state.wizard_step = 0  # End wizard
            st.rerun()

# Main content based on stage
if st.session_state.stage == 'upload':
    st.header("📁 Step 1: Upload Your Dataset")
    # Add tooltip example
    st.markdown("""Upload your file here <span class="tooltip">ℹ️<span class="tooltiptext">Supported files: CSV for simple tables, Excel for spreadsheets, JSON for structured data. If unsure, start with CSV!</span></span>.""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your CSV, Excel, or JSON file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Supported formats: CSV, Excel (.xlsx, .xls), JSON"
        )
        
        if uploaded_file is not None:
            try:
                # Load the data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.session_state.df = df
                st.session_state.cleaner = AgenticDataCleaner(df)
                
                st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
                
                # Show preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Background analysis
                with st.spinner("🔍 Analyzing your data in the background..."):
                    target_detection = st.session_state.cleaner.detect_target_column()
                    st.session_state.target_detection = target_detection
                
                # Show detection results
                if target_detection['top_candidate']:
                    top = target_detection['top_candidate']
                    
                    st.markdown('<div class="reasoning-box">', unsafe_allow_html=True)
                    st.markdown("### 🤖 My Detection")
                    st.markdown(f"**I detected a potential target column:** `{top['column']}`")
                    st.markdown(f"**Confidence Score:** {top['score']}/100")
                    st.markdown("**Reasoning:**")
                    for reason in top['reasons']:
                        st.markdown(f"- {reason}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("➡️ Continue to Target Selection", type="primary", use_container_width=True):
                    st.session_state.stage = 'target'
                    st.session_state.wizard_step = 3
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}. Please check the file format and try again.")
                st.info("Tip: Ensure the file isn't corrupted or too large. If issues persist, try converting to CSV.")

# Together We can

elif st.session_state.stage == 'target':
    st.header("🎯 Step 2: Target Column Selection")
    
    df = st.session_state.df
    cleaner = st.session_state.cleaner
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show AI detection
        if 'target_detection' in st.session_state and st.session_state.target_detection['top_candidate']:
            top = st.session_state.target_detection['top_candidate']
            
            st.markdown('<div class="reasoning-box">', unsafe_allow_html=True)
            st.markdown("### 🤖 My Recommendation")
            st.markdown(f"**Target Column:** `{top['column']}`")
            st.markdown(f"**Confidence:** {top['score']}/100")
            st.markdown(f"**Unique Values:** {top['unique_count']}")
            st.markdown(f"**Data Type:** {top['dtype']}")
            st.markdown("**Why I chose this:**")
            for reason in top['reasons']:
                st.markdown(f"- {reason}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # User selection
        st.subheader("🎯 Confirm or Select Target Column")
        
        default_target = st.session_state.target_detection['top_candidate']['column'] if 'target_detection' in st.session_state else None
        default_index = list(df.columns).index(default_target) if default_target in df.columns else 0
        
        selected_target = st.selectbox(
            "Select your target column:",
            options=df.columns,
            index=default_index,
            help="This is the column you want to predict (your Y variable)"
        )
        
        st.session_state.target_column = selected_target
        
        # Show column preview
        st.subheader(f"📊 Preview of '{selected_target}'")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Unique Values", int(df[selected_target].nunique()))
        with col_b:
            st.metric("Missing", int(df[selected_target].isna().sum()))
        with col_c:
            st.metric("Data Type", str(df[selected_target].dtype))
        
        st.write("**Sample values:**", df[selected_target].head(10).tolist())
        
        # Check if user agrees with AI
        if default_target and selected_target != default_target:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown(f"⚠️ **Note:** You selected `{selected_target}` but I recommended `{default_target}`.")
            st.markdown("That's totally fine! Just making sure you're certain about your choice.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("✅ Confirm and Run Sanity Check", type="primary", use_container_width=True):
            st.session_state.stage = 'sanity'
            st.rerun()
    
    with col2:
        st.info("""
        ### 🎯 What is a Target Column?
        
        The target is what you want to predict:
        - **Classification:** Categories (Survived: 0/1)
        - **Regression:** Numbers (House Price: $200k)
        
        ### ✅ Good Targets
        - Few unique values (2-20)
        - Named like "target", "label"
        - What you want to predict
        
        ### ❌ Bad Targets
        - ID columns (unique per row)
        - Names (too many categories)
        - Constant values (no variation)
        """)

elif st.session_state.stage == 'sanity':
    st.header("✅ Step 3: Sanity Check")
   
    cleaner = st.session_state.cleaner
    target_column = st.session_state.target_column
   
    with st.spinner(f"🔍 Running sanity checks on '{target_column}'..."):
        sanity_result = cleaner.sanity_check_target(target_column)
        cleaner.target_column = target_column
   
    if sanity_result['valid']:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### ✅ All Checks Passed!")
        st.markdown(f"Your target column `{target_column}` looks good for ML!")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.markdown("### ❌ Issues Detected")
        for error in sanity_result['errors']:
            st.markdown(f"- {error}")
        st.markdown('</div>', unsafe_allow_html=True)
   
    # Show detailed checks
    st.subheader("📋 Detailed Quality Checks")
    
    # We'll use a container to group these visually
    with st.container():
        for check in sanity_result['checks']:
            # Skip pure info logs if they don't add value to the checklist
            if check['status'] == 'info' and not st.session_state.explain_mode:
                continue

            # Create a card-like layout for each check
            with st.expander(f"{check['icon']} {check['check']}", expanded=(check['status'] in ['error', 'warning'])):
                col_status, col_text = st.columns([1, 8])
                
                with col_status:
                    if check['status'] == 'success':
                        st.success("Pass")
                    elif check['status'] == 'warning':
                        st.warning("Warn")
                    elif check['status'] == 'error':
                        st.error("Fail")
                
                with col_text:
                    st.markdown(f"**Observation:** Missing values will be displayed on the next page under: Exploratory Data Analysis")
                    
                    # If the message is a dictionary (like value counts), show it as a nice table
                    if isinstance(check.get('details'), dict):
                        st.table(pd.DataFrame.from_dict(check['details'], orient='index', columns=['Value']))
                
                # Explain Mode integration inside the check
                if st.session_state.explain_mode:
                    st.info(f"💡 **Why this check matters:** {get_check_explanation(check['check'])}")

    st.markdown("---")
   
    # Warnings
    if sanity_result['warnings']:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Warnings")
        for warning in sanity_result['warnings']:
            st.markdown(f"- {warning}")
        st.markdown("*These won't stop you, but consider addressing them.*")
        st.markdown('</div>', unsafe_allow_html=True)
   
    st.subheader("📊 Summary")
    try:
        cols = st.columns(4)
        cols[0].metric("Total Rows", int(sanity_result['summary']['total_rows']), help="Total number of data entries in your dataset.")
        cols[1].metric("Target Missing", int(sanity_result['summary']['target_missing_count']), help="How many blanks in your target column—aim for low!")
        cols[2].metric("Target Unique Values", int(sanity_result['summary']['target_unique_count']), help="Distinct values in target—few means classification, many means regression.")
        cols[3].metric("Problem Type", sanity_result['problem_type'].replace('_', ' ').title(), help="Detected ML task type based on your target.")
        
        if st.session_state.explain_mode:
            st.info("**Learning Tip:** If missing values are high, we can impute them later. Unique values help decide if you're classifying (e.g., yes/no) or predicting numbers.")
    except KeyError as ke:
        st.warning(f"Missing summary key: {str(ke)}. Using defaults—check engine output.")
        # Fallback metrics if needed
        cols = st.columns(4)
        cols[0].metric("Total Rows", len(cleaner.df))
        cols[1].metric("Target Missing", "N/A")
        cols[2].metric("Target Unique Values", "N/A")
        cols[3].metric("Problem Type", sanity_result.get('problem_type', 'Unknown').replace('_', ' ').title())
   
    # Navigation
    col1, col2 = st.columns(2)
   
    with col1:
        if st.button("⬅️ Back to Target Selection", use_container_width=True):
            st.session_state.stage = 'target'
            st.rerun()
   
    with col2:
        if st.button("➡️ Proceed to EDA", type="primary", use_container_width=True):
            st.session_state.stage = 'eda'
            st.rerun()

elif st.session_state.stage == 'eda':
    st.header("📈 Step 4: Exploratory Data Analysis")
    
    cleaner = st.session_state.cleaner
    df = st.session_state.df
    target_column = st.session_state.target_column
    
    # Run background analysis
    with st.spinner("🔍 Analyzing your entire dataset..."):
        analysis = cleaner.analyze_background()
        st.session_state.analysis = analysis
    
    # EDA Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Missing Values", "📈 Correlations", "⚡ Outliers"])
    
    with tab1:
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", int(analysis['shape'][0]))
        col2.metric("Columns", int(analysis['shape'][1]))
        col3.metric("Missing Cols", int(len(analysis['missing_values'])))
        col4.metric("Numeric Cols", int(len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])))
        
        st.subheader("📋 Column Data Types")
        dtype_df = pd.DataFrame(list(analysis['dtypes'].items()), columns=['Column', 'Data Type'])
        st.dataframe(dtype_df, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Missing Values Analysis")
        
        if analysis['missing_values']:
            missing_data = []
            for col, info in analysis['missing_values'].items():
                missing_data.append({
                    'Column': col,
                    'Missing Count': info['count'],
                    'Missing %': f"{info['percentage']:.1f}%",
                    'Severity': info['severity'].upper()
                })
            
            missing_df = pd.DataFrame(missing_data)
            
            # Color code by severity
            def color_severity(val):
                if val == 'HIGH':
                    return 'background-color: #f8d7da'
                elif val == 'MEDIUM':
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #d4edda'
            
            st.dataframe(
                missing_df.style.applymap(color_severity, subset=['Severity']),
                use_container_width=True
            )
            
            # Visualization
            fig = px.bar(
                missing_df,
                x='Column',
                y='Missing Count',
                title="Missing Values by Column",
                color='Severity',
                color_discrete_map={'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")
    
    with tab3:
        st.subheader("📈 Feature Correlations with Target")
        
        if analysis['correlations']:
            corr_data = sorted(analysis['correlations'].items(), key=lambda x: abs(x[1]), reverse=True)
            
            corr_df = pd.DataFrame(corr_data, columns=['Feature', 'Correlation'])
            corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
            
            st.dataframe(corr_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                corr_df,
                x='Feature',
                y='Correlation',
                title=f"Correlations with '{target_column}'",
                color='Correlation',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            strong_corr = [f for f, c in corr_data if abs(c) > 0.5]
            if strong_corr:
                st.info(f"💡 **Strong correlations found:** {', '.join(strong_corr)}")
        else:
            st.warning("⚠️ No numeric features to correlate with target")
    
    with tab4:
        st.subheader("⚡ Outlier Detection")
        
        if analysis['outliers']:
            outlier_data = []
            for col, info in analysis['outliers'].items():
                outlier_data.append({
                    'Column': col,
                    'Outlier Count': info['count'],
                    'Outlier %': f"{info['percentage']:.1f}%",
                    'Lower Bound': f"{info['bounds']['lower']:.2f}",
                    'Upper Bound': f"{info['bounds']['upper']:.2f}"
                })
            
            outlier_df = pd.DataFrame(outlier_data)
            st.dataframe(outlier_df, use_container_width=True)
        else:
            st.success("✅ No significant outliers detected!")
    
    # Navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Back to Sanity Check", use_container_width=True):
            st.session_state.stage = 'sanity'
            st.rerun()
    
    with col2:
        if st.button("➡️ Generate Cleaning Plan", type="primary", use_container_width=True):
            with st.spinner("🤖 Creating intelligent cleaning plan..."):
                plan = cleaner.generate_cleaning_plan()
                st.session_state.cleaning_plan = plan
            st.session_state.stage = 'cleaning'
            st.rerun()

elif st.session_state.stage == 'cleaning':
    cleaner = st.session_state.cleaner
    plan = st.session_state.cleaning_plan
    
    if not plan:
        st.warning("⚠️ No cleaning steps generated. Click below to generate a plan.")
        if st.button("🤖 Generate Cleaning Plan"):
            try:  # Wrap for robustness
                with st.spinner("Creating plan..."):
                    plan = cleaner.generate_cleaning_plan()
                    st.session_state.cleaning_plan = plan
                st.rerun()
            except Exception as e:
                st.error(f"Error generating plan: {str(e)}. Please check your data and try again.")
    else:
        st.subheader(f"🤖 I've analyzed your data and recommend {len(plan)} cleaning steps:")
        
        # Group steps by action
        from collections import defaultdict
        grouped_steps = defaultdict(list)
        for step in plan:
            grouped_steps[step.action].append(step)
        
        # Display steps with NEW: Previews, overrides, explanations
        for action, steps in grouped_steps.items():
            with st.expander(f"🔧 {action.upper()} ({len(steps)} steps)", expanded=True):
                for step in steps:
                    st.markdown(f"### Step {step.step_id}: {step.target}")
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**📝 Reasoning:** {step.reasoning} <span class='tooltip'>ℹ️<span class='tooltiptext'>This explains why we suggest this step. Confidence: {step.confidence*100:.0f}% means how sure we are it's helpful.</span></span>")
                        st.markdown(f"**💥 Impact:** {step.impact}")
                        st.markdown(f"**🎯 Confidence:** {step.confidence * 100:.0f}%")
                        
                        if step.alternatives:
                            with st.expander("🔄 Alternatives & Overrides"):
                                for alt in step.alternatives:
                                    st.markdown(f"- **{alt['method']}**: {alt['reason']}")
                                # Override dropdown
                                current_method = step.metadata.get('method', step.action)
                                override_options = [alt['method'] for alt in step.alternatives]
                                selected_override = st.selectbox("Choose a different method?", ['Keep original'] + override_options, key=f"override_{step.step_id}")
                                if selected_override != 'Keep original':
                                    st.session_state.step_overrides[step.step_id] = {'method': selected_override}
                                    st.info(f"Overridden to {selected_override}. We'll apply this when executing.")
                        
                        # Preview button
                        if st.button("👀 Preview Changes", key=f"preview_{step.step_id}"):
                            preview_df = cleaner.preview_step(step)
                            st.markdown("---")
                            with st.container():
                                st.subheader("Sample Comparison")
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("🔍 **Before**")
                                    st.dataframe(cleaner.df.head(5)[[step.target]])
                                with c2:
                                    st.markdown("✨ **After**")
                                    if step.action == 'drop':
                                        st.info("🗑️ Column removed")
                                    else:
                                        st.dataframe(preview_df[[step.target]].head(5))
                            st.markdown("---")
                        
                        # Explain mode content
                        if st.session_state.explain_mode:
                            st.markdown("---")
                            st.markdown("**Learning Tip:**")
                            if step.action == 'impute':
                                st.write("Imputing means filling in missing blanks. Like guessing a puzzle piece from the picture around it—helps keep your data complete!")
                            elif step.action == 'drop':
                                st.write("Dropping removes unhelpful parts. Think tidying your room: get rid of junk to focus on what's useful.")
                            # Anyone can add more links
                            st.markdown("[Learn more about data cleaning basics](https://www.dataquest.io/blog/data-cleaning/)") 
                    
                    with col2:
                        checked = st.checkbox(
                            "Approve",
                            value=True,
                            key=f"step_{step.step_id}"
                        )
                        if checked and step.step_id not in st.session_state.selected_steps:
                            st.session_state.selected_steps.append(step.step_id)
                        elif not checked and step.step_id in st.session_state.selected_steps:
                            st.session_state.selected_steps.remove(step.step_id)
                    
                    st.divider()
        
        # Execute button (updated to handle overrides)
        st.subheader("🚀 Execute Cleaning")
        
        selected_count = len([s for s in plan if st.session_state.get(f"step_{s.step_id}", True)])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Steps", int(len(plan)))
        col2.metric("Selected", int(selected_count))
        col3.metric("Skipped", int(len(plan) - selected_count))
        
        if st.button(f"✅ Execute {selected_count} Selected Steps", type="primary", use_container_width=True):
            try:  # Wrap execution for error handling
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                executed = 0
                for i, step in enumerate(plan):
                    if st.session_state.get(f"step_{step.step_id}", True):
                        # Apply override if set
                        override = st.session_state.step_overrides.get(step.step_id)
                        if override:
                            step.metadata['method'] = override['method']  # Update step for execution
                        
                        status_text.text(f"Executing step {step.step_id}/{len(plan)}: {step.action} {step.target}")
                        
                        result_df, message = cleaner.execute_step(step)
                        st.session_state.df = result_df
                        
                        executed += 1
                        progress_bar.progress((i + 1) / len(plan))
                
                status_text.text(f"✅ Completed! Executed {executed} steps.")
                st.session_state.stage = 'done'
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"Error during execution: {str(e)}. Skipping to recovery—check the step details.")
                # Rollback or log

elif st.session_state.stage == 'done':
    st.header("✨ Cleaning Complete!")
    
    cleaner = st.session_state.cleaner
    summary = cleaner.get_cleaning_summary()
    
    st.success(f"🎉 Successfully cleaned your data!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Rows", int(summary['original_shape'][0]))
    col2.metric("Current Rows", int(summary['current_shape'][0]), delta=int(summary['rows_changed']))
    col3.metric("Original Cols", int(summary['original_shape'][1]))
    col4.metric("Current Cols", int(summary['current_shape'][1]), delta=int(summary['columns_changed']))
    
    # Beginner-friendly insights
    st.subheader("🔍 What Changed & Why It Matters")
    for insight in summary['insights']:
        st.markdown(f"- {insight}")
    if st.session_state.explain_mode:
        st.info("These changes make your data 'cleaner' for analysis or ML—fewer errors mean better results! If you're new, think of it as preparing ingredients before cooking.")
    
    # Show cleaned data
    st.subheader("📊 Cleaned Data Preview")
    st.dataframe(st.session_state.df.head(20), use_container_width=True)

    # OUTLIER VERIFICATION SECTION ---
    st.markdown("---")
    st.subheader("🛡️ Outlier Reduction Verification")
    
    # Get numeric columns that exist in both original and cleaned data
    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
    original_numeric_cols = st.session_state.cleaner.original_df.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = [c for c in numeric_cols if c in original_numeric_cols]

    if common_cols:
        selected_col = st.selectbox("Select a column to compare distributions:", common_cols)
        
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("**Before (Original)**")
            fig_before = px.box(st.session_state.cleaner.original_df, y=selected_col, 
                                points="all", color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig_before, use_container_width=True)
            
        with col_plot2:
            st.markdown("**After (Cleaned)**")
            fig_after = px.box(st.session_state.df, y=selected_col, 
                               points="all", color_discrete_sequence=['#4ECDC4'])
            st.plotly_chart(fig_after, use_container_width=True)
            
        st.info("💡 **How to read this:** Dots outside the 'whiskers' are outliers. If your cleaning plan involved dropping or clipping outliers, the 'After' plot should show a tighter distribution.")
    else:
        st.write("No numeric columns available for outlier comparison.")
    # ------------------------------------------

    # Download options
    st.markdown("---")
    st.subheader("💾 Download Cleaned Data")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel download
        buffer = io.BytesIO()
        st.session_state.df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        
        st.download_button(
            label="📥 Download as Excel",
            data=buffer,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Show what was done
    with st.expander("📋 View Cleaning Steps Executed"):
        for step in summary['executed_steps']:
            st.markdown(f"**Step {step['step_id']}:** {step['action']} - {step['target']}")
            st.markdown(f"*{step['reasoning']}*")
            st.divider()
    
    # Start over
    if st.button("🔄 Clean Another Dataset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>
    🤖 Agentic Data Cleaner v1.1 | Built with Streamlit & AI Reasoning<br>
    Made with ❤️ for Data Scientists
    </small>
</div>
""", unsafe_allow_html=True)
