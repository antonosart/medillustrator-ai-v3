# CSS Fix Script για Readable Text
import re

# Read current file
with open('app_v3_langgraph.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the CSS section
old_css = r'st\.markdown\("""[\s\S]*?</style>[\s\S]*?""", unsafe_allow_html=True\)'

new_css = '''st.markdown("""
    <style>
    /* ============================================
       COMPLETE THEME FIX - READABLE TEXT
       ============================================ */
    
    /* Force light theme globally */
    :root {
        color-scheme: light !important;
    }
    
    /* Main app background - LIGHT */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background-color: #f5f7fa !important;
        color: #1a1a1a !important;
    }
    
    /* All text elements - DARK for readability */
    body, 
    .stApp,
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"],
    p, h1, h2, h3, h4, h5, h6, span, div, label {
        color: #1a1a1a !important;
    }
    
    /* Sidebar - Light with dark text */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    /* Main content area - Light background */
    .main .block-container {
        background-color: #ffffff !important;
        padding: 2rem !important;
        border-radius: 10px !important;
    }
    
    /* Header - Keep blue with white text */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    
    .main-header * {
        color: #ffffff !important;
    }
    
    /* Status indicators with good contrast */
    .status-indicator {
        padding: 0.75rem 1.5rem !important;
        border-radius: 25px !important;
        font-weight: 600 !important;
        text-align: center !important;
        margin: 1rem 0 !important;
        font-size: 1.1rem !important;
    }
    
    .status-ready {
        background: #e3f2fd !important;
        color: #0d47a1 !important;
        border: 2px solid #1976d2 !important;
    }
    
    .status-processing {
        background: #fff3e0 !important;
        color: #e65100 !important;
        border: 2px solid #f57c00 !important;
    }
    
    .status-complete {
        background: #e8f5e9 !important;
        color: #1b5e20 !important;
        border: 2px solid #388e3c !important;
    }
    
    .status-error {
        background: #ffebee !important;
        color: #b71c1c !important;
        border: 2px solid #d32f2f !important;
    }
    
    /* Metric cards - Light with dark text */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        color: #1a1a1a !important;
    }
    
    [data-testid="metric-container"] {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Info boxes and cards */
    .metric-card {
        background: #ffffff !important;
        padding: 1.5rem !important;
        border-radius: 8px !important;
        border-left: 4px solid #007bff !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        margin: 1rem 0 !important;
        color: #1a1a1a !important;
    }
    
    /* Streamlit native components */
    .stButton > button {
        background-color: #007bff !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        background-color: #0056b3 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Tabs - Light with dark text */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    
    /* Input fields - Light with dark text */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #ced4da !important;
        border-radius: 6px !important;
    }
    
    /* File uploader - Light */
    [data-testid="stFileUploader"] {
        background-color: #f8f9fa !important;
        border: 2px dashed #007bff !important;
        border-radius: 8px !important;
        padding: 2rem !important;
    }
    
    [data-testid="stFileUploader"] * {
        color: #1a1a1a !important;
    }
    
    /* Expanders - Light */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: #1a1a1a !important;
        border-radius: 6px !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #007bff !important;
    }
    
    /* Checkbox and radio - Dark text */
    .stCheckbox label,
    .stRadio label {
        color: #1a1a1a !important;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        color: #1a1a1a !important;
    }
    
    /* Data frames and tables */
    .dataframe {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    .dataframe th {
        background-color: #f8f9fa !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        color: #1a1a1a !important;
    }
    
    /* Markdown content */
    .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Code blocks - Dark theme OK here */
    code {
        background-color: #2d2d2d !important;
        color: #f8f8f2 !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
    }
    
    pre {
        background-color: #2d2d2d !important;
        color: #f8f8f2 !important;
        padding: 1rem !important;
        border-radius: 6px !important;
    }
    
    /* Scrollbars - Light */
    ::-webkit-scrollbar {
        width: 10px !important;
        height: 10px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1 !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888 !important;
        border-radius: 5px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555 !important;
    }
    
    /* Tooltips */
    [data-testid="stTooltipIcon"] {
        color: #007bff !important;
    }
    
    /* Footer */
    footer {
        background-color: #f8f9fa !important;
        color: #1a1a1a !important;
        border-top: 1px solid #dee2e6 !important;
        padding: 1rem !important;
    }
    
    /* Ensure all links are visible */
    a {
        color: #007bff !important;
    }
    
    a:hover {
        color: #0056b3 !important;
        text-decoration: underline !important;
    }
    
    /* Image containers */
    [data-testid="stImage"] {
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        background-color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)'''

# Replace
content = re.sub(old_css, new_css, content, flags=re.DOTALL)

# Write back
with open('app_v3_langgraph.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ CSS fix applied successfully!")
print("📝 Changes:")
print("  • Light theme forced globally")
print("  • All text now dark (#1a1a1a) on light backgrounds")
print("  • Status indicators with high contrast")
print("  • Metrics and cards with readable colors")
print("  • Blue header preserved with white text")
