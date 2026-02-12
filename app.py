import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def strip_indent(text):
    """Aggressively strip leading whitespace from every line to avoid Markdown code blocks."""
    if not text: return ""
    return "\n".join([line.lstrip() for line in text.split("\n")])

# Set page config
st.set_page_config(
    page_title="Analisis Ketersediaan Data",
    page_icon="📋",
    layout="wide",
)

# Premium CSS
st.markdown("""
<style>
    .report-card {
        background: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #1e3a8a;
        color: #1e293b;
    }
    .report-card h4 {
        color: #1e3a8a;
        margin-top: 0;
    }
    .report-card p {
        color: #334155;
    }
    .report-card b, .report-card i {
        color: #1e293b;
    }
    .status-ready { color: #059669; font-weight: bold; }
    .status-warning { color: #d97706; font-weight: bold; }
    .status-error { color: #dc2626; font-weight: bold; }
    .rule-box {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #d0e7ff;
        margin-bottom: 30px;
        color: #1e40af;
    }
    .rule-box h3, .rule-box b {
        color: #1e3a8a;
    }
    h1, h2, h3 { color: #1e3a8a; }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def detect_outliers(df, column='Price', threshold=2):
    """Detect outliers using Z-score (Standard Deviation)."""
    if df.empty or len(df) < 3: return []
    mean = df[column].mean()
    std = df[column].std()
    if std == 0: return []
    z_scores = (df[column] - mean) / std
    outliers = df[np.abs(z_scores) > threshold]
    return outliers

# Configuration for Exogenous Variables
EXOGENOUS_CONFIG = {
    'BESI BETON': ['Billet', 'Kurs CNY', 'Kurs JISDOR'],
    'BAJA PROFIL': ['Billet', 'Kurs CNY', 'Kurs JISDOR'], # Assume similar for steel
    'DEFAULT': ['Kurs JISDOR']
}

def generate_pdf_report(summary_df, results, year_range, selected_major, working_df):
    """Generate PDF report with analysis summary and statistics."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=30, leftMargin=30,
                           topMargin=30, bottomMargin=30)
    
    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    # Title
    title = Paragraph("📋 LAPORAN ANALISIS KETERSEDIAAN DATA VENDOR", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Report Info
    report_date = datetime.now().strftime("%d %B %Y, %H:%M")
    info_text = f"""
    <b>Tanggal Laporan:</b> {report_date}<br/>
    <b>Periode Analisis:</b> {year_range[0]} - {year_range[1]}<br/>
    <b>Major Item:</b> {', '.join(selected_major)}<br/>
    """
    elements.append(Paragraph(info_text, normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary Statistics
    elements.append(Paragraph("RINGKASAN STATISTIK", heading_style))
    
    summary_stats = [
        ['Metrik', 'Nilai'],
        ['Total Sub-Material', str(working_df['SubMaterial'].nunique())],
        ['Total Vendor', str(working_df['Vendor'].nunique())],
        ['Total Transaksi', str(len(working_df))],
    ]
    
    stats_table = Table(summary_stats, colWidths=[3*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(stats_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Status Breakdown
    elements.append(Paragraph("STATUS KETERSEDIAAN DATA", heading_style))
    
    status_counts = {'READY': 0, 'PROGRESSING': 0, 'NOT READY': 0}
    for res in results:
        status_counts[res['status']] = status_counts.get(res['status'], 0) + 1
    
    status_data = [
        ['Status', 'Jumlah Sub-Material', 'Persentase'],
        ['READY', str(status_counts['READY']), f"{status_counts['READY']/len(results)*100:.1f}%"],
        ['PROGRESSING', str(status_counts['PROGRESSING']), f"{status_counts['PROGRESSING']/len(results)*100:.1f}%"],
        ['NOT READY', str(status_counts['NOT READY']), f"{status_counts['NOT READY']/len(results)*100:.1f}%"],
    ]
    
    status_table = Table(status_data, colWidths=[2*inch, 2*inch, 1.5*inch])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(status_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Analysis Table (Consolidated)
    elements.append(Paragraph("TABEL ANALISIS KONSOLIDASI", heading_style))
    
    # Prepare table data - split into multiple pages if needed
    table_data = [['Sub-Material', 'Status', 'Coverage (%)', 'Total Data', 'Outliers', 'Avg Price']]
    
    for _, row in summary_df.iterrows():
        table_data.append([
            str(row['Sub-Material'])[:30],  # Truncate long names
            str(row['Status']),
            str(row['Coverage (%)']),
            str(row['Total Data']),
            str(row['Outliers']),
            f"{row['Avg Price']:,.0f}"
        ])
    
    # Create table with appropriate column widths
    col_widths = [2.2*inch, 1*inch, 0.8*inch, 0.8*inch, 0.7*inch, 1*inch]
    analysis_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(analysis_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    elements.append(Paragraph("REKOMENDASI", heading_style))
    
    if len(working_df) < 50:
        recommendation = "⚠️ Data masih sangat sedikit. Disarankan untuk menambah rentang tahun atau data historis agar analisis lebih akurat."
    else:
        recommendation = "✅ Volume data cukup baik untuk analisis awal. Silakan cek detail ketersediaan per bulan untuk insight lebih mendalam."
    
    elements.append(Paragraph(recommendation, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Footer
    footer_text = f"<i>Laporan ini dibuat secara otomatis oleh Sistem Analisis Ketersediaan Data Vendor</i>"
    elements.append(Paragraph(footer_text, normal_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def render_exogenous_input(var_name, full_date_range):
    """Helper to render input section for a single exogenous variable."""
    st.markdown(f"#### 📥 Input Data: {var_name}")
    
    with st.container():
        tab1, tab2 = st.tabs(["📁 Upload CSV", "✍️ Manual Entry"])
        
        with tab1:
            file = st.file_uploader(f"Upload CSV {var_name}", type=['csv'], key=f"up_{var_name}")
            if file:
                try:
                    # Expecting Date and Price columns
                    ex_df = pd.read_csv(file)
                    if len(ex_df.columns) >= 2:
                        # Normalize Date
                        ex_df.iloc[:, 0] = pd.to_datetime(ex_df.iloc[:, 0], dayfirst=True, errors='coerce')
                        ex_df.columns.values[0] = 'Date'
                        
                        # Normalize Price (2nd col)
                        col_val = ex_df.columns[1]
                        ex_df[col_val] = pd.to_numeric(ex_df[col_val].astype(str).str.replace(',', ''), errors='coerce')
                        
                        # Resample to monthly
                        ex_df = ex_df.set_index('Date').resample('MS').mean().reset_index()
                        st.success(f"✅ Loaded {len(ex_df)} rows for {var_name}")
                        return ex_df[['Date', ex_df.columns[1]]] # Return Date + Value
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    
        with tab2:
            st.info(f"Masukkan data {var_name} per bulan.")
            # Pre-fill dates
            manual_data = pd.DataFrame({'Date': full_date_range, 'Price': [0.0]*len(full_date_range)})
            # We must convert to datetime for data_editor to handle it as dates, 
            # but usually it's better to keep as object or format nicely.
            # standard st.column_config.DateColumn prefers date objects.
            manual_data['Date'] = manual_data['Date'].dt.date
            
            edited = st.data_editor(
                manual_data,
                column_config={
                    "Date": st.column_config.DateColumn("Bulan", disabled=True, format="YYYY-MM"),
                    "Price": st.column_config.NumberColumn(f"Harga {var_name}", required=True)
                },
                key=f"edit_{var_name}",
                num_rows="fixed",
                hide_index=True,
                use_container_width=True
            )
            
            # Check if user input anything (sum > 0 as proxy)
            if edited['Price'].sum() != 0:
                edited['Date'] = pd.to_datetime(edited['Date'])
                return edited
                
    return None

@st.cache_data
def load_data(uploaded_file):
    """Load data strictly from the uploaded file with column validation."""
    if uploaded_file is None:
        return pd.DataFrame(), []
        
    try:
        encodings = ['utf-8', 'latin1', 'cp1252', 'utf-16']
        df = None
        
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except:
                continue
        
        if df is None:
            return pd.DataFrame(), []

        column_mapping = {
            'Name of Supplier': 'Vendor',
            'Document Date': 'Date',
            'Short Text': 'SubMaterial',
            'Major Item': 'MajorItem',
            'Order Quantity': 'Quantity',
            'Net Price': 'Price',
            'Currency': 'Currency' # Added for filtering
        }
        
        # Validation: Check for required original SAP columns
        required_sap_cols = list(column_mapping.keys())
        missing_cols = [col for col in required_sap_cols if col not in df.columns]
        
        if missing_cols:
            return pd.DataFrame(), missing_cols

        # Process and Rename
        df = df.rename(columns=column_mapping)
        
        # Basic Cleaning
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')
        
        # Filter IDR Only
        if 'Currency' in df.columns:
            df = df[df['Currency'] == 'IDR']
        
        # Drop NA
        df = df.dropna(subset=['Vendor', 'MajorItem', 'Date'])
        
        # String Cleaning (Strip Whitespace & Upper)
        str_cols = ['Vendor', 'SubMaterial', 'MajorItem']
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().str.upper()
        
        # Vendor Specific Cleaning
        df['Vendor'] = df['Vendor'].apply(clean_string)
        
        # Ensure no empty strings
        df = df[df['Vendor'] != ""]
        
        # Add Period column
        df['MonthPeriod'] = df['Date'].dt.to_period('M')
        
        return df, []
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), []

def clean_string(s):
    if pd.isna(s): return ""
    # Remove leading numbers and separators (e.g. "10001 - PT ABC" -> "PT ABC")
    s = str(s).strip()
    s = pd.Series(s).str.replace(r'^\d+[\s\-\.]*', '', regex=True).iloc[0]
    # Replace multiple spaces with one and upper case
    s = ' '.join(str(s).split()).upper()
    return s

def analyze_readiness(data, target_months):
    # Check coverage within target period
    target_start = target_months[0].to_timestamp()
    target_end = target_months[-1].to_timestamp(how='end')
    target_count = len(target_months)
    
    sub_data_period = data[(data['Date'] >= target_start) & (data['Date'] <= target_end)]
    months_present = set(sub_data_period['MonthPeriod'].unique())
    target_months_set = set(target_months)
    
    missing_months = sorted(list(target_months_set - months_present))
    coverage_pct = (len(months_present) / target_count) * 100
    
    missing_str = ", ".join([m.strftime('%b %Y') for m in missing_months])
    period_str = f"{target_months[0].strftime('%Y')} - {target_months[-1].strftime('%Y')}"
    
    if len(missing_months) == 0:
        return "READY", f"Data lengkap di setiap bulan ({period_str}).", "status-ready", coverage_pct, ""
    elif coverage_pct >= 50:
        return "PROGRESSING", f"Cakupan data {coverage_pct:.1f}%.", "status-warning", coverage_pct, missing_str
    else:
        return "NOT READY", f"Cakupan data sangat rendah ({coverage_pct:.1f}%).", "status-error", coverage_pct, missing_str

@st.cache_data
def get_ready_summary(_df, _target_months):
    """Identify sub-materials that have at least one vendor with 100% coverage."""
    target_start = _target_months[0].to_timestamp()
    target_end = _target_months[-1].to_timestamp(how='end')
    target_count = len(_target_months)
    
    # Filter target period
    period_df = _df[(_df['Date'] >= target_start) & (_df['Date'] <= target_end)]
    if period_df.empty: return []
    
    # Group by SubMaterial and Vendor to check individual coverage
    vendor_coverage = period_df.groupby(['SubMaterial', 'Vendor'])['MonthPeriod'].nunique().reset_index()
    
    # Vendors with 100% coverage
    qualified = vendor_coverage[vendor_coverage['MonthPeriod'] == target_count]
    
    # Unique sub-materials that have at least one such vendor
    ready_sub_materials = qualified['SubMaterial'].unique().tolist()
    return ready_sub_materials

def get_analysis_results(working_df, target_months):
    """Core logic to analyze all sub-materials in the working dataframe."""
    sub_materials = sorted(working_df['SubMaterial'].unique())
    target_count = len(target_months)
    results = []
    
    for sub in sub_materials:
        sub_data = working_df[working_df['SubMaterial'] == sub]
        status, desc, css_class, coverage, missing = analyze_readiness(sub_data, target_months)
        
        # Outlier Detection
        outliers = detect_outliers(sub_data)
        has_outliers = len(outliers) > 0
        
        # Vendor Completeness
        target_start = target_months[0].to_timestamp()
        target_end = target_months[-1].to_timestamp(how='end')
        period_data = sub_data[(sub_data['Date'] >= target_start) & (sub_data['Date'] <= target_end)]
        
        vendors_in_sub = period_data['Vendor'].unique()
        vendor_stats = []
        for v in vendors_in_sub:
            v_data = period_data[period_data['Vendor'] == v]
            v_month_count = v_data['MonthPeriod'].nunique()
            vendor_stats.append({'name': v, 'count': v_month_count})
            
        # Sort vendors by count
        vendor_stats = sorted(vendor_stats, key=lambda x: x['count'], reverse=True)
        complete_vendors = [v['name'] for v in vendor_stats if v['count'] == target_count]
        top_vendors = vendor_stats[:3]
        
        # Price Stats
        avg_p = sub_data['Price'].mean()
        min_p = sub_data['Price'].min()
        max_p = sub_data['Price'].max()
        
        results.append({
            'sub': sub,
            'status': status,
            'class': css_class,
            'desc': desc,
            'coverage': coverage,
            'missing': missing,
            'has_outliers': has_outliers,
            'outlier_count': len(outliers),
            'avg_price': avg_p,
            'min_price': min_p,
            'max_price': max_p,
            'total_data': len(sub_data),
            'vendors': complete_vendors,
            'top_vendors': top_vendors,
            'sub_data': sub_data # Store df for plotting/download
        })
    return results

def render_analysis_config(df):
    """Renders the configuration section and returns selected filters."""
    with st.expander("⚙️ Konfigurasi Analisis", expanded=True):
        c1, c2 = st.columns(2)
        
        with c1:
            current_year = datetime.now().year
            def_val = st.session_state.get('year_range', (2022, 2024))
            year_range = st.slider(
                "Pilih Rentang Tahun Analisis",
                min_value=2018, max_value=current_year,
                value=def_val,
                key="config_year_range"
            )
            
        with c2:
            major_options = sorted(df['MajorItem'].unique())
            def_major = st.session_state.get('selected_major', [])
            # Ensure default is valid
            def_major = [m for m in def_major if m in major_options]
            
            selected_major = st.multiselect(
                "Pilih Major Item", 
                major_options, 
                default=def_major,
                key="config_selected_major"
            )
        
    # Generate Target Months immediately
    start_date = f"{year_range[0]}-01"
    end_date = f"{year_range[1]}-12"
    target_months = pd.period_range(start=start_date, end=end_date, freq='M')
    
    # Store in session state for persistence
    st.session_state['year_range'] = year_range
    st.session_state['selected_major'] = selected_major
    st.session_state['target_months'] = target_months
    
    return selected_major, target_months, year_range

def home_view():
    st.header("🏠 Beranda: Setup Analisis")
    
    # 1. Rules (simplified from original main)
    with st.expander("📌 Aturan & Langkah Penggunaan (Klik untuk pelajari)"):
        st.markdown(strip_indent("""
            **Langkah-langkah:**
            1. Masukkan file CSV SAP.
            2. Tentukan Rentang Tahun & Major Item di bawah.
            3. Klik tombol **Mulai Analisis** untuk melihat hasil.
        """))

    # 2. Upload
    st.markdown("### 📂 1. Input Data SAP")
    uploaded_file = st.file_uploader("Drop atau pilih file CSV SAP Anda di sini", type=["csv"])
    
    if uploaded_file:
        df, missing_cols = load_data(uploaded_file)
        if missing_cols:
            st.error(f"⚠️ **File Tidak Valid!** Kolom hilang: `{', '.join(missing_cols)}`")
            return
            
        if not df.empty:
            # Persist data
            st.session_state['data_raw'] = df
            st.success("✅ File berhasil dimuat!")
            
    # Check if data exists in session
    if 'data_raw' not in st.session_state:
        st.info("👋 Silakan unggah file CSV untuk memulai.")
        return

    df = st.session_state['data_raw']
    
    # 3. Proceed to Dashboard
    st.markdown("---")
    if st.button("🚀 Lanjut ke Dashboard", type="primary", use_container_width=True):
        st.session_state['page'] = 'Dashboard' 
        st.rerun()

def main():
    st.title("📋 Sistem Analisis Ketersediaan Data Vendor")

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.header("Jelajahi Menu")
        # Use session state to control selection if set progammatically
        if 'page' not in st.session_state:
            st.session_state['page'] = 'Beranda'
            
        # Map radio to session state is tricky if we want bi-directional.
        # Simplest: Just use radio as source of truth, but allow 'NEXT' button to override?
        # Streamlit radio index defaults to 0. 
        # We can use keys.
        
        # Define menu options
        options = ["Beranda", "Dashboard", "Preparation Data", "Feature Engineering"]
        
        # Determine index based on current page
        try:
            idx = options.index(st.session_state['page'])
        except:
            idx = 0
            
        selected_page = st.radio("Menu", options, index=idx, key="nav_radio")
        
        # Sync: If radio changed, update page. 
        # But if Button changed page, we want radio to update (handled by index=idx above + rerun)
        if selected_page != st.session_state['page']:
            st.session_state['page'] = selected_page
            st.rerun()
            
    # --- PAGE ROUTING ---
    if st.session_state['page'] == "Beranda":
        home_view()
        
    elif st.session_state['page'] == "Dashboard":
        # Check requirements
        # Check requirements
        if 'data_raw' not in st.session_state:
            st.warning("⚠️ Data belum siap. Silakan kembali ke **Beranda** untuk setup.")
            if st.button("Ke Beranda"):
                st.session_state['page'] = 'Beranda'
                st.rerun()
        else:
            # Load from session
            df = st.session_state['data_raw']
            
            # Render Configuration (Top of Dashboard)
            majors, months, years = render_analysis_config(df)
            
            # Result View now contains Dashboard + Detail logic
            # We can use tabs or just stick to the previous Dashboard -> Detail flow?
            # User said "Menu Result". 
            # Let's keep the sub-navigation (Dashboard vs Detail) internal to this view?
            # Or simplified: Result = Dashboard View. Detail View is a sub-feature.
            
            # Let's use tabs for Result: Ringkasan vs Detail
            tab_res1, tab_res2 = st.tabs(["📊 Ringkasan (Dashboard)", "📑 Detail Report"])
            
            with tab_res1:
                dashboard_view(df, majors, months, years)
            with tab_res2:
                detail_view(df, majors, months, years)

    elif st.session_state['page'] == "Preparation Data":
        if 'data_raw' not in st.session_state:
             st.warning("⚠️ Silakan kembali ke Beranda untuk upload data.")
        else:
            preprocessing_view(st.session_state['data_raw'])

    elif st.session_state['page'] == "Feature Engineering":
        if 'fe_input_data' not in st.session_state:
            st.warning("⚠️ Data belum siap. Silakan selesaikan **Preparation Data** terlebih dahulu.")
            if st.button("↩️ Ke Preparation Data"):
                st.session_state['page'] = 'Preparation Data'
                st.rerun()
        else:
            feature_engineering_view(st.session_state['fe_input_data'])

def dashboard_view(df, selected_major, target_months, year_range):
    st.header("📊 Dashboard Ringkasan")
    
    if not selected_major:
        st.info("👈 Silakan pilih **Major Item** di sidebar untuk melihat ringkasan.")
        return

    working_df = df[df['MajorItem'].isin(selected_major)]
    
    # Summary Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Sub-Material Terpilih", f"{working_df['SubMaterial'].nunique()}")
    c2.metric("Total Vendor", f"{working_df['Vendor'].nunique()}")
    c3.metric("Total Transaksi", f"{len(working_df)}")
    
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Distribusi Sub-Material")
        sub_counts = working_df['SubMaterial'].value_counts().reset_index()
        sub_counts.columns = ['SubMaterial', 'Count']
        fig_pie = px.pie(sub_counts, values='Count', names='SubMaterial', title='Frekuensi Data per Sub-Material')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.subheader("Top 5 Vendor")
        vendor_counts = working_df['Vendor'].value_counts().head(5).reset_index()
        vendor_counts.columns = ['Vendor', 'Count']
        fig_bar = px.bar(vendor_counts, x='Count', y='Vendor', orientation='h', title='Top 5 Vendor Teraktif')
        st.plotly_chart(fig_bar, use_container_width=True)

    # Consolidated Table
    st.markdown("---")
    st.subheader("📊 Tabel Rangkuman Analisis Konsolidasi")
    
    # Run Analysis
    results = get_analysis_results(working_df, target_months)
    target_count = len(target_months)
    
    summary_data = []
    for res in results:
        top_v_str = ", ".join([f"{v['name']} ({v['count']}/{target_count})" for v in res['top_vendors']])
        summary_data.append({
            'Sub-Material': res['sub'],
            'Status': res['status'],
            'Coverage (%)': round(res['coverage'], 1),
            'Total Data': res['total_data'],
            'Outliers': res['outlier_count'],
            'Avg Price': round(res['avg_price'], 2) if not pd.isna(res['avg_price']) else 0,
            'Min Price': round(res['min_price'], 2) if not pd.isna(res['min_price']) else 0,
            'Max Price': round(res['max_price'], 2) if not pd.isna(res['max_price']) else 0,
            'Vendors (Complete)': ", ".join(res['vendors']) if res['vendors'] else "None",
            'Top 3 Vendors': top_v_str if top_v_str else "None"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Download Buttons
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        csv_summary = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Ringkasan (.CSV)",
            data=csv_summary,
            file_name=f"Summary_Analisis_{year_range[0]}_{year_range[1]}.csv",
            mime='text/csv',
            use_container_width=True
        )
    
    with col_dl2:
        # Generate PDF
        pdf_buffer = generate_pdf_report(summary_df, results, year_range, selected_major, working_df)
        st.download_button(
            label="📑 Download Laporan PDF",
            data=pdf_buffer,
            file_name=f"Laporan_Analisis_{year_range[0]}_{year_range[1]}.pdf",
            mime='application/pdf',
            use_container_width=True
        )

    st.markdown("### 💡 Saran Tindak Lanjut")
    if len(working_df) < 50:
        st.warning("Data masih sangat sedikit. Disarankan untuk menambah rentang tahun atau data historis agar analisis lebih akurat.")
    else:
        st.success("Volume data cukup baik untuk analisis awal. Silakan cek detail ketersediaan per bulan.")

    st.markdown("---")
    st.markdown("### 🚀 Langkah Selanjutnya")
    
    col_nav1, col_nav2 = st.columns([2, 1])
    with col_nav1:
        st.info("💡 Gunakan tab **Detail Report** di atas untuk melihat analisis mendalam per sub-material, atau lanjutkan ke tahap preprocessing untuk membersihkan dan mempersiapkan data.")
    
    with col_nav2:
        if st.button("➡️ Lanjut ke Preprocessing", type="primary", use_container_width=True):
            st.session_state['page'] = 'Preparation Data'
            st.rerun()

def detail_view(df, selected_major, target_months, year_range):
    # Back button removed
        
    if not selected_major:
        st.warning("Pilih Major Item di sidebar.")
        return

    working_df = df[df['MajorItem'].isin(selected_major)]
    st.subheader(f"📑 Detail Report: {', '.join(selected_major)}")
    
    # Run Analysis
    results = get_analysis_results(working_df, target_months)
    target_count = len(target_months)
    
    for res in results:
        # Display Card
        with st.container():
            missing_info = f'<p style="color: #dc2626; font-size: 0.9em;"><b>⚠️ Bulan Kosong:</b> {res["missing"]}</p>' if res["missing"] else ""
            outlier_info = f'<p style="color: #991b1b; background-color: #fef2f2; padding: 5px;">🕵️ <b>Outlier:</b> {res["outlier_count"]} data.</p>' if res["has_outliers"] else ""
            vendor_html = f'<p style="color: #059669;"><b>✅ Vendor Lengkap:</b> {", ".join(res["vendors"])}</p>' if res["vendors"] else '<p style="color: #64748b;">ℹ️ Tidak ada vendor 100% lengkap.</p>'

            st.markdown(strip_indent(f"""
                <div class="report-card">
                    <h4>📦 {res['sub']}</h4>
                    <p>Status: <span class="{res['class']}">{res['status']}</span> (Cov: {res['coverage']:.1f}%)</p>
                    {outlier_info}
                    {missing_info}
                    {vendor_html}
                </div>
            """), unsafe_allow_html=True)
            
            # Action Buttons per Card
            c1, c2, c3 = st.columns(3)
            with c1:
                with st.expander("📄 Lihat Data"):
                    st.dataframe(res['sub_data'], use_container_width=True)
            with c2:
                csv = res['sub_data'].to_csv(index=False).encode('utf-8')
                st.download_button(f"📥 Download Data", csv, f"{res['sub']}.csv", "text/csv", key=f"dl_{res['sub']}")
            with c3:
                if st.button(f"📈 Generate Plot", key=f"plot_{res['sub']}"):
                    fig = px.line(res['sub_data'], x='Date', y='Price', color='Vendor', title=f"Time Series: {res['sub']}")
                    st.plotly_chart(fig, use_container_width=True)

def preprocessing_view(df):
    st.header("🛠️ Preprocessing Pipeline")

    # 1. Selection
    st.markdown("### 1. Pilih Data untuk Diproses")
    
    # helper for recommendations
    def get_top3(dframe, col):
        return dframe[col].value_counts().head(3).index.tolist()

    # Level 1: Major Item
    col_major = st.columns(1)[0]
    major_opts = sorted(df['MajorItem'].unique())
    
    with col_major:
        selected_major = st.selectbox("1. Pilih Major Item", major_opts)
        top_major = get_top3(df, 'MajorItem')
        st.caption(f"💡 **Rekomendasi (Top 3 Data Terbanyak):** {', '.join(top_major)}")

    if not selected_major: return

    # Level 2: Sub-Material
    major_df = df[df['MajorItem'] == selected_major]
    
    # Sorting option for Sub-Material
    col_sub_header, col_sub_sort = st.columns([2, 1])
    with col_sub_header:
        st.markdown("#### 2. Pilih Sub-Material")
    with col_sub_sort:
        sort_sub = st.radio(
            "Urutkan:",
            ["Abjad (A-Z)", "Jumlah Data (Terbanyak)"],
            horizontal=True,
            key="sort_sub",
            label_visibility="collapsed"
        )
    
    # Apply sorting based on selection
    if sort_sub == "Abjad (A-Z)":
        sub_opts = sorted(major_df['SubMaterial'].unique())
    else:
        # Sort by data count descending
        sub_counts = major_df['SubMaterial'].value_counts()
        sub_opts = sub_counts.index.tolist()
    
    col_sub = st.columns(1)[0]
    with col_sub:
        selected_sub = st.selectbox(
            "Sub-Material",
            sub_opts,
            label_visibility="collapsed"
        )
        top_sub = get_top3(major_df, 'SubMaterial')
        st.caption(f"💡 **Rekomendasi (Top 3):** {', '.join(top_sub)}")
        
    if not selected_sub: return

    # Level 3: Vendor
    sub_df = major_df[major_df['SubMaterial'] == selected_sub]
    
    # Sorting option for Vendor
    col_vendor_header, col_vendor_sort = st.columns([2, 1])
    with col_vendor_header:
        st.markdown("#### 3. Pilih Vendor")
    with col_vendor_sort:
        sort_vendor = st.radio(
            "Urutkan:",
            ["Abjad (A-Z)", "Jumlah Data (Terbanyak)"],
            horizontal=True,
            key="sort_vendor",
            label_visibility="collapsed"
        )
    
    # Apply sorting based on selection
    if sort_vendor == "Abjad (A-Z)":
        vendor_opts = sorted(sub_df['Vendor'].unique())
    else:
        # Sort by data count descending
        vendor_counts = sub_df['Vendor'].value_counts()
        vendor_opts = vendor_counts.index.tolist()
    
    col_vendor = st.columns(1)[0]
    with col_vendor:
        selected_vendor = st.selectbox(
            "Vendor",
            vendor_opts,
            label_visibility="collapsed"
        )
        top_vendor = get_top3(sub_df, 'Vendor')
        st.caption(f"💡 **Rekomendasi (Top 3):** {', '.join(top_vendor)}")
        
    if not selected_vendor:
        st.warning("Silakan pilih Vendor.")
        return

    # Filter Data (Final)
    raw_df = sub_df[sub_df['Vendor'] == selected_vendor].copy()
    raw_df = raw_df[['Date', 'Price']].sort_values('Date')
    
    if raw_df.empty:
        st.error("❌ Data tidak ditemukan untuk kombinasi Sub-Material dan Vendor ini.")
        return
    
    st.markdown("---")
    st.markdown("### 2. Automated Health Check & Cleaning")
    
    # Check 1: Duplicates & Feature Selection
    duplicates = raw_df.duplicated(subset=['Date']).sum()
    st.info(f"✅ **Feature Selection**: Hanya kolom `Date` dan `Price` yang digunakan.\n\n"
            f"🔍 **Duplicate Check**: Ditemukan {duplicates} data tanggal ganda. (Akan di-aggregate via Mean)")

    # Check 2: Monthly Aggregation
    # Resample to Monthly Start 'MS' and take mean price
    idx_df = raw_df.set_index('Date')
    monthly_df = idx_df.resample('MS').mean().reset_index()
    monthly_df.rename(columns={'Price': 'Price_Raw'}, inplace=True)
    
    # Check 3: Missing Values (Gaps in months)
    # Create full period range
    full_idx = pd.date_range(start=monthly_df['Date'].min(), end=monthly_df['Date'].max(), freq='MS')
    monthly_df = monthly_df.set_index('Date').reindex(full_idx).reset_index()
    monthly_df.rename(columns={'index': 'Date'}, inplace=True)
    
    missing_count = monthly_df['Price_Raw'].isna().sum()
    
    # Check 4: Outliers (IQR)
    Q1 = monthly_df['Price_Raw'].quantile(0.25)
    Q3 = monthly_df['Price_Raw'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = monthly_df[(monthly_df['Price_Raw'] < lower_bound) | (monthly_df['Price_Raw'] > upper_bound)]
    outlier_count = len(outliers)
    
    # Check 5: Variance (Std Dev)
    std_dev = monthly_df['Price_Raw'].std()
    mean_val = monthly_df['Price_Raw'].mean()
    cv = (std_dev / mean_val) if mean_val != 0 else 0 # Coefficient of Variation
    
    # Display Health Stats
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Bulan", len(monthly_df))
    s2.metric("Missing Values", f"{missing_count}")
    s3.metric("Outliers", f"{outlier_count}")
    s4.metric("Volatilitas (CV)", f"{cv:.2%}", help="Coefficient of Variation (Std/Mean). Semakin tinggi, semakin fluktuatif harga.")
    
    st.markdown("---")
    st.markdown("### 3. Interactive Handling")
    
    # Handling Options
    c_handle1, c_handle2 = st.columns(2)
    
    with c_handle1:
        st.subheader("🔧 Missing Value Handling")
        missing_opt = st.radio(
            "Metode Penanganan Missing Value:",
            ["Biarkan (Drop)", "Forward Fill", "Backward Fill", "Mean Imputation", "Linear Interpolation"],
            index=4
        )
        
    with c_handle2:
        st.subheader("🔧 Outlier Handling")
        outlier_opt = st.radio(
            "Metode Penanganan Outlier:",
            ["Biarkan", "Hapus (Drop)", "Cap (Winsorize) ke Batas Atas/Bawah"],
            index=2
        )
        
    # Apply Handling
    clean_df = monthly_df.copy()
    clean_df['Price_Clean'] = clean_df['Price_Raw']
    
    # 1. Handle Outliers first (optional sequence, but usually clean valid data then impute)
    # Actually better to impute then handle outliers? Or handle outliers then impute?
    # Let's handle outliers in existing data first.
    if outlier_opt == "Hapus (Drop)":
        clean_df.loc[(clean_df['Price_Clean'] < lower_bound) | (clean_df['Price_Clean'] > upper_bound), 'Price_Clean'] = np.nan
    elif outlier_opt == "Cap (Winsorize) ke Batas Atas/Bawah":
        clean_df['Price_Clean'] = clean_df['Price_Clean'].clip(lower=lower_bound, upper=upper_bound)
        
    # 2. Handle Missing (Now includes outliers converted to NaN if dropped)
    if missing_opt == "Biarkan (Drop)":
        clean_df = clean_df.dropna(subset=['Price_Clean'])
    elif missing_opt == "Forward Fill":
        clean_df['Price_Clean'] = clean_df['Price_Clean'].ffill()
    elif missing_opt == "Backward Fill":
        clean_df['Price_Clean'] = clean_df['Price_Clean'].bfill()
    elif missing_opt == "Mean Imputation":
        mean_val = clean_df['Price_Clean'].mean()
        clean_df['Price_Clean'] = clean_df['Price_Clean'].fillna(mean_val)
    elif missing_opt == "Linear Interpolation":
        clean_df['Price_Clean'] = clean_df['Price_Clean'].interpolate(method='linear')

    # --- Comparison Chart (Raw vs Clean) inside Interactive Handling ---
    st.markdown("#### 📈 Perbandingan: Raw vs Clean")
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=monthly_df['Date'], y=monthly_df['Price_Raw'], mode='lines+markers', name='Raw (Monthly Agg)', line=dict(color='red', dash='dot')))
    fig_compare.add_trace(go.Scatter(x=clean_df['Date'], y=clean_df['Price_Clean'], mode='lines+markers', name='Clean Price', line=dict(color='green')))
    fig_compare.update_layout(title='Raw vs Clean Price', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_compare, use_container_width=True)

    # Visualization
    st.markdown("---")
    st.markdown("### 4. Integrasi Data Eksternal (Exogenous)")
    
    # Check config based on Major Item
    # We need to find the Major Item for the selected Sub-Material
    # Since we filtered earlier, we can get it from df
    current_major = df[df['SubMaterial'] == selected_sub]['MajorItem'].iloc[0]
    required_vars = EXOGENOUS_CONFIG.get(current_major, EXOGENOUS_CONFIG['DEFAULT'])
    
    st.info(f"Untuk **{current_major}**, data pendukung yang disarankan: **{', '.join(required_vars)}**")
    
    # Dictionary to store external dataframes
    if 'exogenous_data' not in st.session_state:
        st.session_state['exogenous_data'] = {}
        
    exogenous_data = st.session_state['exogenous_data']
    
    # -------------------------------------------------------------
    # 2024-05-24: FIX MANUAL ENTRY DATE RANGE
    # Use the full analysis year range configured in Dashboard
    # -------------------------------------------------------------
    year_range = st.session_state.get('year_range', (2022, 2024))
    start_date = f"{year_range[0]}-01-01"
    end_date = f"{year_range[1]}-12-31"
    idx_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Dropdown Selection
    exo_options = required_vars + ["Lainnya (Custom)"]
    selected_exo = st.selectbox("Pilih Data Pendukung untuk Diinput:", exo_options)
    
    # Handle Input
    if selected_exo == "Lainnya (Custom)":
        custom_name = st.text_input("Nama Variabel Baru:", key="custom_exo_name")
        if custom_name:
            df_ex = render_exogenous_input(custom_name, idx_range)
            if df_ex is not None:
                st.session_state['exogenous_data'][custom_name] = df_ex
                st.success(f"✅ Data {custom_name} tersimpan sementara.")
    else:
        df_ex = render_exogenous_input(selected_exo, idx_range)
        if df_ex is not None:
            st.session_state['exogenous_data'][selected_exo] = df_ex
            st.success(f"✅ Data {selected_exo} tersimpan sementara.")
            
    # Show collected data status
    if st.session_state['exogenous_data']:
        st.markdown(f"**Data Tersimpan:** {', '.join(st.session_state['exogenous_data'].keys())}")
        if st.button("Hapus Semua Data Eksternal"):
            st.session_state['exogenous_data'] = {}
            st.rerun()

    # Merge Logic
    final_df = clean_df.copy()
    
    if st.session_state['exogenous_data']:
        st.success(f"🔗 Menggabungkan {len(st.session_state['exogenous_data'])} variabel eksternal...")
        for var_name, df_ex in st.session_state['exogenous_data'].items():
            # Merge on Date (both should be datetime start of month)
            # Ensure keys are same type
            final_df['Date'] = pd.to_datetime(final_df['Date'])
            df_ex['Date'] = pd.to_datetime(df_ex['Date'])
            
            # Left join to keep main data structure
            final_df = pd.merge(final_df, df_ex, on='Date', how='left')
            # Rename value column to var_name
            # Check if column name exists to avoid duplication errors or handle overwrite
            if var_name in final_df.columns:
                 # If overwrite needed, drop first? Or update? Merge usually suffixes.
                 # Let's drop existing if present before merge or handle suffixes. 
                 # Simplest: just rename the new column
                 pass
            
            # The df_ex has columns ['Date', 'ValueColName']
            # We need to rename 'ValueColName' to var_name. 
            # But wait, df_ex from render_exogenous_input might have 'Price' or similar.
            # Let's inspect render_exogenous_input return. It returns ['Date', col_val]
            # so we rename col_val to var_name.
            val_col = [c for c in df_ex.columns if c != 'Date'][0]
            final_df.rename(columns={val_col: var_name}, inplace=True)
            
        # Display Correlation
        st.markdown(f"#### 📊 Korelasi Harga vs Data Eksternal")
        corr_cols = ['Price_Clean'] + list(st.session_state['exogenous_data'].keys())
        corr_matrix = final_df[corr_cols].corr()
        
        # Show Heatmap or just simple correlation with Price
        price_corr = corr_matrix[['Price_Clean']].sort_values(by='Price_Clean', ascending=False)
        st.dataframe(price_corr, use_container_width=True) # Removed .style to fix error
        
        # Plot Comparison
        var_to_plot = st.selectbox("Pilih Variabel untuk Overlay:", list(st.session_state['exogenous_data'].keys()))
        
        fig_dual = go.Figure()
        # Main Price (Left Axis)
        fig_dual.add_trace(go.Scatter(x=final_df['Date'], y=final_df['Price_Clean'], name='Harga Material', line=dict(color='blue')))
        # Exogenous (Right Axis)
        if var_to_plot in final_df.columns:
             fig_dual.add_trace(go.Scatter(x=final_df['Date'], y=final_df[var_to_plot], name=var_to_plot, line=dict(color='orange', dash='dot'), yaxis='y2'))
        
        fig_dual.update_layout(
            title=f"Trend Harga vs {var_to_plot}",
            yaxis=dict(title="Harga Material"),
            yaxis2=dict(title=var_to_plot, overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h')
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    st.markdown("---")
    st.markdown("### 5. Visualisasi & Validasi (Final Dataset)")
    
    # Combined Time Series: Raw + Clean + Exogenous
    fig_final = go.Figure()
    
    # Raw Price
    fig_final.add_trace(go.Scatter(
        x=monthly_df['Date'], y=monthly_df['Price_Raw'],
        mode='lines+markers', name='Raw Price',
        line=dict(color='#ef4444', dash='dot', width=1),
        marker=dict(size=4), opacity=0.6
    ))
    
    # Final Clean Price
    fig_final.add_trace(go.Scatter(
        x=final_df['Date'], y=final_df['Price_Clean'],
        mode='lines+markers', name='Final Clean Price',
        line=dict(color='#22c55e', width=2),
        marker=dict(size=5)
    ))
    
    # Exogenous Variables (on secondary y-axis)
    exo_colors = ['#f97316', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    exo_keys = list(st.session_state.get('exogenous_data', {}).keys())
    for i, var_name in enumerate(exo_keys):
        if var_name in final_df.columns:
            color = exo_colors[i % len(exo_colors)]
            fig_final.add_trace(go.Scatter(
                x=final_df['Date'], y=final_df[var_name],
                mode='lines', name=var_name,
                line=dict(color=color, dash='dash', width=1.5),
                yaxis='y2'
            ))
    
    layout_args = dict(
        title='Time Series: Raw vs Clean Price' + (' + Exogenous' if exo_keys else ''),
        xaxis_title='Date',
        yaxis=dict(title='Harga Material'),
        legend=dict(x=0, y=1.15, orientation='h'),
        hovermode='x unified'
    )
    if exo_keys:
        layout_args['yaxis2'] = dict(title='Variabel Eksogen', overlaying='y', side='right')
    
    fig_final.update_layout(**layout_args)
    st.plotly_chart(fig_final, use_container_width=True)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.caption("Preview Dataset Akhir (Termasuk Variabel Eksternal):")
        st.dataframe(final_df, use_container_width=True)
        
    with col_dl2:
        # Download the FINAL merged dataframe
        csv_final = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Final Dataset (.CSV)",
            data=csv_final,
            file_name=f"Ready_For_Model_{selected_sub}_{selected_vendor}.csv",
            mime="text/csv"
        )
        
        st.info("Dataset ini sudah bersih, teragregasi bulanan, dan memiliki variabel eksternal (jika ada). Siap untuk modelling.")
        
        if st.button("🚀 Lanjut ke Feature Engineering"):
              st.session_state['fe_input_data'] = final_df.copy()
              st.session_state['page'] = 'Feature Engineering'
              st.rerun()

def feature_engineering_view(df):
    st.header("⚙️ Feature Engineering")
    st.caption("Buat fitur-fitur baru dari dataset yang sudah bersih untuk meningkatkan performa model forecasting.")

    # Show source data info
    cols_info = [c for c in df.columns if c != 'Date']
    st.info(f"📊 Dataset masuk: **{len(df)} baris**, kolom: `{', '.join(cols_info)}`")

    # Identify price column and exogenous columns
    price_col = 'Price_Clean' if 'Price_Clean' in df.columns else [c for c in df.columns if 'price' in c.lower()][0] if any('price' in c.lower() for c in df.columns) else df.columns[1]
    exo_cols = [c for c in df.columns if c not in ['Date', 'Price_Raw', price_col, 'MonthPeriod']]

    fe_df = df[['Date', price_col]].copy()
    fe_df['Date'] = pd.to_datetime(fe_df['Date'])
    # Also keep exogenous in working df
    for ec in exo_cols:
        if ec in df.columns:
            fe_df[ec] = df[ec].values

    new_features = []

    # ── Section 1: Lag Features ──────────────────────────────────
    with st.expander("📐 1. Lag Features", expanded=True):
        st.markdown("Membuat kolom harga dari bulan-bulan sebelumnya sebagai fitur prediktif.")
        lag_options = list(range(1, 13))
        selected_lags = st.multiselect(
            "Pilih Lag (bulan):",
            lag_options,
            default=[1, 2, 3],
            key="fe_lags"
        )
        if selected_lags:
            for lag in sorted(selected_lags):
                col_name = f"Lag_{lag}"
                fe_df[col_name] = fe_df[price_col].shift(lag)
                new_features.append(col_name)
            st.success(f"✅ Ditambahkan {len(selected_lags)} lag features: {', '.join([f'Lag_{l}' for l in sorted(selected_lags)])}")

    # ── Section 2: Rolling Statistics ────────────────────────────
    with st.expander("📊 2. Rolling Statistics", expanded=True):
        st.markdown("Menghitung statistik jendela geser (moving window) untuk menangkap tren dan volatilitas.")
        c1, c2 = st.columns(2)
        with c1:
            window_sizes = st.multiselect(
                "Window Size (bulan):",
                [3, 6, 12],
                default=[3, 6],
                key="fe_windows"
            )
        with c2:
            rolling_stats = st.multiselect(
                "Statistik:",
                ["Mean", "Std", "Min", "Max"],
                default=["Mean", "Std"],
                key="fe_rolling_stats"
            )
        if window_sizes and rolling_stats:
            for w in sorted(window_sizes):
                roller = fe_df[price_col].rolling(window=w, min_periods=1)
                if "Mean" in rolling_stats:
                    col_name = f"Rolling_Mean_{w}"
                    fe_df[col_name] = roller.mean()
                    new_features.append(col_name)
                if "Std" in rolling_stats:
                    col_name = f"Rolling_Std_{w}"
                    fe_df[col_name] = roller.std()
                    new_features.append(col_name)
                if "Min" in rolling_stats:
                    col_name = f"Rolling_Min_{w}"
                    fe_df[col_name] = roller.min()
                    new_features.append(col_name)
                if "Max" in rolling_stats:
                    col_name = f"Rolling_Max_{w}"
                    fe_df[col_name] = roller.max()
                    new_features.append(col_name)
            st.success(f"✅ Ditambahkan {len(window_sizes) * len(rolling_stats)} rolling features.")

    # ── Section 3: Rate of Change ────────────────────────────────
    with st.expander("📈 3. Rate of Change", expanded=False):
        st.markdown("Mengukur perubahan harga dari bulan ke bulan, baik dalam persentase maupun absolut.")
        roc_options = st.multiselect(
            "Pilih Tipe:",
            ["Pct Change (%)", "Absolute Diff"],
            default=["Pct Change (%)"],
            key="fe_roc"
        )
        if "Pct Change (%)" in roc_options:
            fe_df['Pct_Change'] = fe_df[price_col].pct_change() * 100
            new_features.append('Pct_Change')
        if "Absolute Diff" in roc_options:
            fe_df['Abs_Diff'] = fe_df[price_col].diff()
            new_features.append('Abs_Diff')
        if roc_options:
            st.success(f"✅ Ditambahkan: {', '.join(roc_options)}")

    # ── Section 4: Calendar Features ─────────────────────────────
    with st.expander("📅 4. Calendar / Seasonal Features", expanded=False):
        st.markdown("Menambahkan encoding waktu untuk menangkap pola musiman.")
        cal_options = st.multiselect(
            "Pilih Fitur Kalender:",
            ["Month", "Quarter", "Year", "Month_Sin", "Month_Cos"],
            default=["Month", "Quarter"],
            key="fe_calendar"
        )
        if "Month" in cal_options:
            fe_df['Month'] = fe_df['Date'].dt.month
            new_features.append('Month')
        if "Quarter" in cal_options:
            fe_df['Quarter'] = fe_df['Date'].dt.quarter
            new_features.append('Quarter')
        if "Year" in cal_options:
            fe_df['Year'] = fe_df['Date'].dt.year
            new_features.append('Year')
        if "Month_Sin" in cal_options:
            fe_df['Month_Sin'] = np.sin(2 * np.pi * fe_df['Date'].dt.month / 12)
            new_features.append('Month_Sin')
        if "Month_Cos" in cal_options:
            fe_df['Month_Cos'] = np.cos(2 * np.pi * fe_df['Date'].dt.month / 12)
            new_features.append('Month_Cos')
        if cal_options:
            st.success(f"✅ Ditambahkan: {', '.join(cal_options)}")

    # ── Section 5: Custom Ratio (Price vs Exogenous) ─────────────
    if exo_cols:
        with st.expander("🔗 5. Rasio Harga vs Variabel Eksogen", expanded=False):
            st.markdown("Membuat fitur rasio antara harga material dan variabel eksogen.")
            selected_ratio_vars = st.multiselect(
                "Pilih variabel eksogen untuk rasio:",
                exo_cols,
                key="fe_ratio_vars"
            )
            for var in selected_ratio_vars:
                col_name = f"Ratio_{price_col}_vs_{var}"
                fe_df[col_name] = fe_df[price_col] / fe_df[var].replace(0, np.nan)
                new_features.append(col_name)
            if selected_ratio_vars:
                st.success(f"✅ Ditambahkan {len(selected_ratio_vars)} rasio features.")
    else:
        st.markdown("---")
        st.caption("ℹ️ Tidak ada variabel eksogen. Bagian Rasio dilewati.")

    # ── Summary & Preview ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Ringkasan Fitur Baru")

    if new_features:
        s1, s2, s3 = st.columns(3)
        s1.metric("Total Fitur Baru", len(new_features))
        s2.metric("Total Kolom Dataset", len(fe_df.columns))
        s3.metric("Total Baris", len(fe_df))

        # NaN info
        nan_counts = fe_df[new_features].isna().sum()
        nan_features = nan_counts[nan_counts > 0]
        if not nan_features.empty:
            with st.expander("⚠️ Fitur dengan NaN (akibat shifting/rolling)"):
                st.dataframe(nan_features.reset_index().rename(columns={'index': 'Feature', 0: 'NaN Count'}), hide_index=True)
                drop_nan = st.checkbox("Drop baris dengan NaN?", value=False, key="fe_drop_nan")
                if drop_nan:
                    fe_df = fe_df.dropna()
                    st.info(f"Dataset setelah drop NaN: **{len(fe_df)} baris**")

        # Correlation Preview
        st.markdown("#### 🔥 Korelasi Fitur Baru vs Harga")
        numeric_cols = [price_col] + [f for f in new_features if f in fe_df.columns]
        corr_data = fe_df[numeric_cols].corr()[[price_col]].drop(price_col).sort_values(by=price_col, ascending=False)
        corr_data.columns = ['Korelasi']

        # Color-coded bar chart
        fig_corr = px.bar(
            corr_data.reset_index().rename(columns={'index': 'Feature'}),
            x='Korelasi', y='Feature', orientation='h',
            color='Korelasi',
            color_continuous_scale='RdYlGn',
            title='Korelasi Fitur vs Harga'
        )
        fig_corr.update_layout(yaxis={'categoryorder': 'total ascending'}, height=max(300, len(new_features) * 28))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Time Series Preview of top features
        st.markdown("#### 📈 Preview Time Series")
        preview_features = st.multiselect(
            "Pilih fitur untuk divisualisasikan:",
            new_features,
            default=new_features[:3] if len(new_features) >= 3 else new_features,
            key="fe_preview_select"
        )
        if preview_features:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=fe_df['Date'], y=fe_df[price_col],
                mode='lines+markers', name=price_col,
                line=dict(color='#2563eb', width=2)
            ))
            colors = ['#f97316', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308', '#10b981']
            for i, feat in enumerate(preview_features):
                fig_ts.add_trace(go.Scatter(
                    x=fe_df['Date'], y=fe_df[feat],
                    mode='lines', name=feat,
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    yaxis='y2'
                ))
            fig_ts.update_layout(
                title='Harga vs Fitur Baru',
                yaxis=dict(title=price_col),
                yaxis2=dict(title='Features', overlaying='y', side='right'),
                legend=dict(x=0, y=1.15, orientation='h'),
                hovermode='x unified'
            )
            st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Belum ada fitur baru yang dipilih. Aktifkan fitur di atas untuk mulai.")

    # ── Final Dataset Preview & Download ─────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Dataset Final (Siap Modelling)")
    st.dataframe(fe_df, use_container_width=True, hide_index=True)

    csv_fe = fe_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset Feature Engineering (.CSV)",
        data=csv_fe,
        file_name="Feature_Engineered_Dataset.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.success("✅ Dataset siap digunakan untuk training model forecasting!")

    
if __name__ == "__main__":
    main()
