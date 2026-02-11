🤖 Aka Cleaner by pR😅D2Be 
pR😅D2Be was founded on the belief that intelligence and innovation are not limited by background, ethinicity, environment or status. Every individual and every system deserves to be Proud To Be.
Aka Cleaner is an intelligent data cleaning tool with human-in-the-loop reasoning.

An open-source tool that doesn't just clean your data, it reasons with you. Aka Cleaner explains its choices, predicts the impact of changes, and requires your approval before modifying a single row.

🌟 Key Features
1. Intelligent Target Detection
Heuristic Engine: Automatically identifies your target/label column.

Explainability: It tells you why it chose that column (cardinality, naming patterns, distribution).

2. Comprehensive Sanity Checks
Validation Suite: Automatically checks for class imbalance, data type consistency, and valid ranges.

Early Warnings: Identifies potential issues before you start the cleaning process.

3. Interactive EDA (Exploratory Data Analysis)
Visual Health Checks: Correlation heatmaps, missing value matrices, and outlier distributions.

Statistical Insight: Uses the IQR method for robust outlier detection.

4. Agentic Reasoning & Mentorship
Impact Analysis: Shows exactly what will happen (e.g., "You will fill 177 missing values").

Mentor Mode: Integrated educational tooltips that explain the "Why" behind data science operations.

🚀 Quick Start
Installation
Bash

# Clone the repository
git clone https://github.com/LOSTBAGWELL/agentic-data-cleaner.git
cd agentic-data-cleaner

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run agentic_cleaner_app.py
Usage Workflow
Upload: Drop in a CSV, Excel, or JSON file.

Detect: Confirm the AI-detected target column.

Verify: Review the automated Sanity Check results.

Plan: Approve, Reject, or Customize the AI-generated cleaning steps.

Export: Download your high-quality, cleaned dataset.

📊 Example Reasoning
AI Target Detection
🤖 AI Observation: "I have detected Survived as the target column (Confidence: 92%). Reasoning: It is a binary integer column (0/1) which is typical for classification tasks. Other columns like Name were rejected due to 100% uniqueness."

Cleaning Recommendation
Step 1: DROP 'Cabin'

Reasoning: 77.1% missing values. Imputation would introduce too much noise.

Impact: Reduces feature set by 1; preserves row count.

Alternative: Impute with "Unknown" category.

🏗️ System Architecture
Modular Design
The project is split into two distinct layers for better maintainability:

agentic_data_cleaner.py: The "Brain." A pure Python engine that performs statistical analysis and manages the state of the cleaning plan.

agentic_cleaner_app.py: The "Face." A Streamlit-based reactive UI that translates engine logic into a user-friendly dashboard.

Data Flow Pipeline
Code snippet

graph LR
    A[Raw Data] --> B{Agentic Engine}
    B --> C[Statistical Analysis]
    B --> D[Step Generation]
    D --> E[User Approval]
    E --> F[Clean Dataset]
🧪 Testing with Titanic Dataset
Python

from agentic_data_cleaner import AgenticDataCleaner
import pandas as pd

# Initialize
df = pd.read_csv('titanic.csv')
cleaner = AgenticDataCleaner(df)

# 1. AI Target detection
target = cleaner.detect_target_column()

# 2. Run background diagnostics
analysis = cleaner.analyze_background()

# 3. Generate cleaning steps
plan = cleaner.generate_cleaning_plan()

# 4. Execute a step with impact tracking
result_df, message = cleaner.execute_step(plan[0])
print(message) 
# Output: "Dropped 'Cabin' due to high missingness (77%)"
📈 Roadmap
[x] Phase 1: Core Engine & Streamlit MVP.

[ ] Phase 2: Advanced Imputation (KNN & Iterative Imputer).

[ ] Phase 3: Automated Feature Engineering suggestions.

[ ] Phase 4: Export cleaning logic as a reusable .py script.

🤝 Contributing & License
We welcome contributions! If you have a new cleaning strategy or a UI improvement, please open a Pull Request.

License: Distributed under the MIT License. See LICENSE for more information.

📧 Contact
KALUBA MUBANGA MORDECAI

GitHub: @LOSTBAGWELL

Email: mubangamordecaikaluba@gmail.com.com

Project Link: https://github.com/LOSTBAGWELL/agentic-data-cleaner

Made with ❤️ to make data science accessible for everyone.
