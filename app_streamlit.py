import streamlit as st
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import urllib.parse  

# Move interpolation setup to top
def setup_hjv_interpolator():
    x_vals = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    y_vals = list(range(80, 111))
    
    # Full z_matrix with 31 rows (y_vals) x 15 columns (x_vals)
    z_matrix = np.array([
    [0, 15.90, 26.50, 35.60, 43.80, 51.50, 59.00, 65.70, 72.40, 79.10, 85.60, 98.80, 111.70, 123.60, 134.20, 142.90],
    [0, 16.15, 26.90, 36.15, 44.50, 52.30, 59.90, 66.70, 73.50, 80.30, 86.95, 100.35, 113.45, 125.50, 136.30, 145.10],
    [0, 16.40, 27.35, 36.70, 45.15, 53.10, 60.75, 67.75, 74.60, 81.55, 88.30, 101.90, 115.20, 127.45, 138.35, 147.35],
    [0, 16.60, 27.75, 37.30, 45.85, 53.90, 61.75, 68.75, 75.75, 82.75, 89.60, 103.40, 116.90, 129.35, 140.45, 149.55],
    [0, 16.85, 28.00, 37.85, 46.50, 54.70, 62.70, 69.80, 76.90, 84.00, 90.95, 104.95, 118.65, 131.30, 142.50, 151.80],
    [0, 17.10, 28.60, 38.40, 47.20, 55.50, 63.60, 70.80, 78.00, 85.20, 92.30, 106.50, 120.40, 133.20, 144.60, 154.00],
    [0, 17.30, 29.00, 38.90, 47.80, 56.25, 64.45, 71.75, 79.05, 86.35, 93.55, 107.95, 122.00, 135.00, 146.55, 156.05],
    [0, 17.55, 29.35, 39.45, 48.45, 57.00, 65.30, 72.70, 80.10, 87.50, 94.80, 109.40, 123.60, 136.75, 148.50, 158.10],
    [0, 17.75, 29.75, 39.95, 49.05, 57.70, 66.20, 73.70, 81.10, 88.60, 96.00, 110.80, 125.20, 138.55, 150.40, 160.20],
    [0, 18.00, 30.10, 40.50, 49.70, 58.45, 67.05, 74.65, 82.15, 89.75, 97.25, 112.25, 126.80, 140.30, 152.35, 162.23],
    [0, 18.20, 30.50, 41.00, 50.30, 59.20, 67.90, 75.60, 83.20, 90.90, 98.50, 113.70, 128.40, 142.10, 154.30, 164.30],
    [0, 18.40, 30.85, 41.50, 50.90, 59.90, 68.70, 76.50, 84.20, 92.00, 99.65, 115.05, 129.95, 143.80, 156.15, 166.25],
    [0, 18.65, 31.20, 41.95, 51.50, 60.60, 69.50, 77.35, 86.20, 93.05, 100.80, 116.40, 131.50, 145.50, 158.00, 168.20],
    [0, 18.85, 31.60, 42.45, 52.10, 61.30, 70.30, 78.50, 87.20, 94.15, 102.00, 117.70, 133.00, 147.20, 159.80, 170.10],
    [0, 19.10, 31.95, 42.90, 52.70, 62.00, 71.10, 79.10, 88.20, 95.20, 103.15, 119.05, 134.55, 148.90, 161.65, 172.05],
    [0, 19.30, 32.30, 43.40, 53.30, 62.70, 71.90, 80.00, 89.10, 96.30, 104.30, 120.40, 136.10, 150.60, 163.50, 174.00],
    [0, 19.50, 32.65, 43.85, 53.85, 63.40, 72.65, 80.85, 90.05, 97.30, 105.40, 121.70, 137.55, 152.20, 165.20, 175.05],
    [0, 19.75, 33.00, 44.30, 54.40, 64.05, 73.40, 81.70, 90.95, 98.35, 106.50, 122.95, 139.00, 153.75, 166.95, 177.70],
    [0, 19.95, 33.30, 44.80, 55.40, 64.75, 74.20, 82.60, 91.90, 99.35, 107.60, 124.25, 140.40, 155.35, 168.65, 179.50],
    [0, 20.20, 33.65, 45.25, 55.55, 65.40, 74.95, 83.45, 91.90, 100.40, 108.70, 125.50, 141.85, 156.90, 170.40, 181.35],
    [0, 20.40, 34.00, 45.70, 56.10, 66.10, 75.70, 84.30, 92.80, 101.40, 109.80, 126.80, 143.30, 158.50, 172.10, 183.20],
    [0, 20.60, 34.30, 46.15, 56.65, 66.70, 76.40, 85.10, 93.70, 102.40, 110.85, 128.00, 144.65, 160.00, 173.75, 184.95],
    [0, 20.75, 34.65, 46.60, 57.20, 67.35, 77.10, 85.90, 94.50, 103.35, 111.90, 129.20, 146.00, 161.55, 175.40, 186.70],
    [0, 20.95, 34.95, 47.00, 57.70, 67.95, 77.85, 86.70, 95.50, 104.35, 113.00, 130.40, 147.40, 163.05, 177.00, 188.50],
    [0, 21.10, 35.30, 47.45, 58.50, 68.60, 78.60, 87.50, 96.40, 105.30, 114.05, 131.60, 148.75, 164.60, 178.65, 190.50],
    [0, 21.30, 35.60, 47.90, 58.80, 69.20, 79.30, 88.30, 97.30, 106.30, 115.10, 132.80, 150.10, 166.10, 180.30, 192.00],
    [0, 21.50, 35.80, 48.30, 59.30, 69.80, 80.00, 89.10, 98.15, 107.20, 116.10, 133.95, 151.40, 167.56, 181.90, 193.70],
    [0, 21.70, 36.25, 48.75, 59.85, 70.40, 80.70, 89.85, 99.00, 108.15, 117.10, 135.10, 152.75, 169.00, 183.45, 195.35],
    [0, 21.90, 36.56, 49.16, 60.35, 71.00, 81.40, 90.66, 99.80, 109.05, 118.10, 136.30, 154.05, 170.50, 185.05, 197.05],
    [0, 22.10, 36.90, 49.60, 60.90, 71.60, 82.10, 91.40, 100.65, 110.00, 119.10, 137.45, 155.40, 171.95, 186.60, 198.70],
    [0, 22.30, 37.20, 50.00, 61.40, 72.20, 82.80, 92.20, 101.50, 110.90, 120.10, 138.60, 156.70, 173.40, 188.20, 200.00],
])

    
    return RegularGridInterpolator((y_vals, x_vals), z_matrix)

# Initialize interpolator at top level
interpolator = setup_hjv_interpolator()

def calculate_hjv_debit(opening_percent, res_level):
    """Calculate HJV debit based on opening percentage and reservoir level"""
    try:
        return float(interpolator([[res_level, opening_percent]])[0])
    except ValueError:
        return 0.0

st.set_page_config(page_title="Perhitungan Debit Sesaat Bendungan Ir. H. Djuanda", layout="wide")
st.title("Perhitungan Debit Sesaat Bendungan Ir. H. Djuanda")

# Create tabs for navigation
tab1, tab2 = st.tabs(["Input Data", "Hasil Perhitungan"])

with tab1:
    st.markdown('''<span style="color:yellow; background-color:black; font-weight:bold">
                Beban di bawah 15 Mw\nTinggi jatuh head\nlihat tabel\n(debit turbin input manual)</span>''', 
                unsafe_allow_html=True)
    
    st.subheader("Data Utama")
    col1, col2 = st.columns(2)
    with col1:
        tma = st.number_input("Tinggi Muka Air Waduk (mdpl)", 
                             value=None,
                             step=0.01, 
                             format="%.2f")
    with col2:
        trc = st.number_input("Tailrace (mdpl)", 
                             value=None,
                             step=0.01, 
                             format="%.2f")
    
    # Calculate head automatically with None check
    tinggi_jatuh = (tma or 0) - (trc or 0)  # Use 0 if None for calculations
    st.info(f"Tinggi Jatuh (head) = {tinggi_jatuh:.2f} m")

    st.subheader("Data Hollow Cone Valve")
    col1, col2 = st.columns(2)
    with col1:
        hjv_kiri = st.number_input("HCV Kiri (%)", 
                                  value=None,
                                  step=1)
        debit_hjv_kiri = calculate_hjv_debit(hjv_kiri, tma)
        st.write(f"Debit HCV Kiri: {debit_hjv_kiri:.2f} m³/det")

    with col2:
        hjv_kanan = st.number_input("HCV Kanan (%)", 
                                   value=None,
                                   step=1)
        debit_hjv_kanan = calculate_hjv_debit(hjv_kanan, tma)
        st.write(f"Debit HCV Kanan: {debit_hjv_kanan:.2f} m³/det")

    # Display total HJV debit with None check
    total_hjv = (debit_hjv_kiri or 0) + (debit_hjv_kanan or 0)  # Use 0 if None
    st.info(f"Total Debit HCV = {total_hjv:.2f} m³/det")

    # Input Beban per Unit section
    st.subheader("Input Beban per Unit")
    unit_list = ["I", "II", "III", "IV", "V", "VI"]
    beban = []

    # Replace the horizontal columns with vertical layout
    for i in range(6):
        beban_i = st.number_input(f"Beban Unit {unit_list[i]} (mW)",
                                 value=None, 
                                 step=0.01,
                                 format="%.2f",
                                 key=f"beban_{i}")
        beban.append(beban_i or 0.0)  # Use 0.0 if None for calculations
    
    # Display total beban
    total_beban = sum(beban)
    st.info(f"Total Beban = {total_beban:.2f} MW")

    
# Rumus R5
R5 = -0.32675 + 3.5945*tinggi_jatuh - 0.0463189*tinggi_jatuh**2 + 0.0001975*tinggi_jatuh**3

# Rumus R6-R11
R = []
for i in range(6):
    denominator = -30857 + 1292.71*tinggi_jatuh - 8.9741*tinggi_jatuh**2 + 0.03682*tinggi_jatuh**3
    if denominator != 0:
        R_i = beban[i]*100*1000/denominator
    else:
        R_i = 0
    R.append(R_i)

# Rumus L6-L11
L = []
for i in range(6):
    poly = -4.532068452 + 0.31155337*R[i] - 0.006520552181*R[i]**2 + 0.0000597737436*R[i]**3 - 0.0000002019124*R[i]**4
    denominator = 9.8 * tinggi_jatuh * R5 * (poly/100)
    if denominator != 0:
        L_i = beban[i]/denominator*1000
    else:
        L_i = 0
    L.append(L_i)

# L12 calculation with modified check
has_low_beban = any(b < 15 and b > 0 for b in beban)  # Only check active units

if has_low_beban:
    st.warning("Ada beban di bawah 15 MW. Silakan input Debit Turbin secara manual.")
    L12 = st.number_input("Input Debit Turbin m³/det", 
                        value=sum(L), 
                        step=0.001,
                        format="%.3f")
else:
    L12 = sum(L)

# L13 calculation with None check
if tma is None or tma <= 106.9:
    L13 = 0
else:
    L13 = 231.2*((tma-106.9)**1.5) + 15.8*((tma-106.9)**2.5)

# L14
L14 = total_hjv  # Use the total HJV debit
# L15
L15 = L12 + L13 + L14

with tab2:
    st.subheader("Hasil Perhitungan Debit")
    
    # Display unit debits vertically
    for i in range(6):
        st.metric(f"Debit Unit {unit_list[i]}", f"{L[i]:,.3f} m³/det")
    
    st.divider()
    
    # Display summary debits vertically
    st.metric("Debit Turbin", f"{L12:,.3f} m³/det")
    st.metric("Debit Limpasan", f"{L13:,.3f} m³/det")
    st.metric("Debit HJV Total", f"{L14:,.3f} m³/det")
    
    st.divider()
    st.metric("Debit Total", f"{L15:,.3f} m³/det", delta=f"{L15-L12:,.3f} m³/det")

    st.divider()
    
    # Get current date and time
    from datetime import datetime
    current_time = datetime.now()
    
    # Indonesian day names mapping
    hari = {
        'Monday': 'Senin',
        'Tuesday': 'Selasa',
        'Wednesday': 'Rabu',
        'Thursday': 'Kamis',
        'Friday': 'Jumat',
        'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }
    
    # Indonesian month names mapping
    bulan = {
        'January': 'Januari',
        'February': 'Februari',
        'March': 'Maret',
        'April': 'April',
        'May': 'Mei',
        'June': 'Juni',
        'July': 'Juli',
        'August': 'Agustus',
        'September': 'September',
        'October': 'Oktober',
        'November': 'November',
        'December': 'Desember'
    }
    
    # Format the message with Indonesian date
    day_en = current_time.strftime('%A')
    month_en = current_time.strftime('%B')
    
    # Count active units
    active_units = sum(1 for b in beban if b > 0)
    active_unit_numbers = [str(i+1) for i, b in enumerate(beban) if b > 0]
    
    st.divider()
    
    # Manual date and time input
    st.subheader("Waktu Pengiriman")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_date = st.date_input(
            "Tanggal",
            value=current_time.date(),
            format="DD/MM/YYYY"
        )
    with col2:
        jam = st.number_input("Jam", min_value=0, max_value=23, value=current_time.hour)
    with col3:
        menit = st.number_input("Menit", min_value=0, max_value=59, value=current_time.minute)
    
    # Format the messages with selected date and handle None values
    whatsapp_message = f"""{hari[day_en]}, {selected_date.strftime('%d')} {bulan[month_en]} {selected_date.strftime('%Y')}
Jam : {jam:02d}:{menit:02d} WIB
Bendungan Ir.H.Djuanda
TMA Waduk : {tma if tma is not None else 0:.2f} mdpl
TMA Tailrace : {trc if trc is not None else 0:.2f} mdpl
Turbin : {active_units} ({','.join(active_unit_numbers)}) Unit
Total Beban : {sum(beban):.2f} MW
Bukaan Hollow jet Valve(HJV)
Kiri : {hjv_kiri if hjv_kiri is not None else 0:.1f}% Kanan : {hjv_kanan if hjv_kanan is not None else 0:.1f}%
Debit Turbin : {L12:.3f} m³/s
Debit Limpasan : {L13:.3f} m³/s
Debit HJV : {L14:.3f} m³/s
Debit total Sesaat : {L15:.3f} m³/s"""
    
    # Create share URL for WhatsApp only
    whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(whatsapp_message)}"

    # Add single share button
    st.divider()
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.markdown(f'''
        <a href="{whatsapp_url}" target="_blank">
            <button style="
                background-color: #25D366;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            ">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M13.601 2.326A7.854 7.854 0 0 0 7.994 0C3.627 0 .068 3.558.064 7.926c0 1.399.366 2.76 1.057 3.965L0 16l4.204-1.102a7.933 7.933 0 0 0 3.79.965h.004c4.368 0 7.926-3.558 7.93-7.93A7.898 7.898 0 0 0 13.6 2.326zM7.994 14.521a6.573 6.573 0 0 1-3.356-.92l-.24-.144-2.494.654.666-2.433-.156-.251a6.56 6.56 0 0 1-1.007-3.505c0-3.626 2.957-6.584 6.591-6.584a6.56 6.56 0 0 1 4.66 1.931 6.557 6.557 0 0 1 1.928 4.66c-.004 3.639-2.961 6.592-6.592 6.592zm3.615-4.934c-.197-.099-1.17-.578-1.353-.646-.182-.065-.315-.099-.445.099-.133.197-.513.646-.627.775-.114.133-.232.148-.43.05-.197-.1-.836-.308-1.592-.985-.59-.525-.985-1.175-1.103-1.372-.114-.198-.011-.304.088-.403.087-.088.197-.232.296-.346.1-.114.133-.198.198-.33.065-.134.034-.248-.015-.347-.05-.099-.445-1.076-.612-1.47-.16-.389-.323-.335-.445-.34-.114-.007-.247-.007-.38-.007a.729.729 0 0 0-.529.247c-.182.198-.691.677-.691 1.654 0 .977.71 1.916.81 2.049.098.133 1.394 2.132 3.383 2.992.47.205.84.326 1.129.418.475.152.904.129 1.246.08.38-.058 1.171-.48 1.338-.943.164-.464.164-.86.114-.943-.049-.084-.182-.133-.38-.232z"/>
                </svg>
                Kirim ke WhatsApp
            </button>
        </a>
        ''', unsafe_allow_html=True)

    # Preview message
    with st.expander("Preview Pesan"):
        st.code(whatsapp_message)
