import streamlit as st

st.set_page_config(page_title="Thông tin tác giả", page_icon="ℹ️",
                   layout="wide", initial_sidebar_state="collapsed")

# ================== NAVIGATION ==================
st.sidebar.header("Navigation Menu")
st.sidebar.page_link("gui_project2.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/intro.py", label="Giới thiệu", icon="📃")
st.sidebar.page_link('pages/suggest_bikes.py', label='Gợi ý xe máy tương tự', icon="⭐")
st.sidebar.page_link('pages/predictprice_byclustering.py', label='Dự đoán giá xe máy cũ', icon="💵")
st.sidebar.page_link("pages/author.py", label="Thông tin tác giả", icon="ℹ️")

# ================== PAGE INFO ==================
st.header("👤 Nhóm tác giả dự án")
st.write("""
**Hồ Thị Quỳnh Như**  
**Nguyễn Văn Cường**  
**Nguyễn Thị Tuyết Anh**  

Giáo viên hướng dẫn: ThS. Khuất Thùy Phương
""")