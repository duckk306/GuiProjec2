import streamlit as st

st.set_page_config(page_title="Gợi ý xe máy tương tự", page_icon="⭐",
                   layout="wide", initial_sidebar_state="collapsed")

# ================== NAVIGATION ==================
st.sidebar.header("Navigation Menu")
st.sidebar.page_link("gui_project2.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/intro.py", label="Giới thiệu", icon="📃")
st.sidebar.page_link('pages/suggest_bikes.py', label='Gợi ý xe máy tương tự', icon="⭐")
st.sidebar.page_link('pages/predictprice_byclustering.py', label='Dự đoán giá xe máy cũ', icon="💵")
st.sidebar.page_link("pages/author.py", label="Thông tin tác giả", icon="ℹ️")

# ---- TITLE ----
st.title("🏍️ Hệ thống gợi ý xe máy và dự đoán giá xe máy cũ")
st.markdown("---")

#st.image("imgs/xe_may.jpg", caption="Xe máy cũ")

# ---- INTRO CONTENT ----
st.markdown("""
## 🌟 Giới thiệu hệ thống

**Chợ Tốt** là một trong những nền tảng mua bán trực tuyến lớn nhất Việt Nam, 
nơi mỗi ngày có hàng ngàn tin đăng về xe máy. Điều này khiến người dùng gặp khó khăn khi:

- Tìm kiếm chiếc xe phù hợp giữa vô số tin đăng.
- Đánh giá xem **mức giá người bán đưa ra có hợp lý hay không**.

Để hỗ trợ trải nghiệm người dùng, hệ thống này được xây dựng với hai chức năng chính:
""")

# ---- FEATURE 1 ----
st.markdown("""
---

## 🚀 1. Gợi ý xe máy tương tự

Hệ thống gợi ý danh sách các xe có đặc điểm tương tự với lựa chọn của người dùng:

- Người dùng chọn thông tin mô tả chiếc xe mong muốn.
- Hệ thống truy vấn và trả về danh sách xe tương tự.
- Có thể tuỳ chọn số lượng xe muốn hiển thị.

""")

# ---- FEATURE 2 ----
st.markdown("""
---

## 💰 2. Dự đoán giá xe máy cũ

Hệ thống hỗ trợ định giá dựa trên các yếu tố như:

- Thương hiệu
- Độ phổ biến
- Giá tham khảo
- Năm sản xuất
- Tình trạng sử dụng  
- Các đặc điểm kỹ thuật khác

Hệ thống áp dụng các kỹ thuật **phân cụm (clustering)** để phân chia xe vào những phân khúc thị trường riêng biệt trước khi dự đoán, giúp mô hình đưa ra mức giá ước lượng **chính xác và phù hợp hơn**.

---
""")

st.info(
    """✨ Hệ thống được xây dựng nhằm hỗ trợ người dùng lựa chọn xe dễ dàng hơn và tham khảo mức giá hợp lý trên thị trường.

    Thực hiện bởi nhóm sinh viên 
        Data Science Class - TTTH ĐH Khoa học Tư nhiên:
        - Nguyễn Thị Tuyết Anh
        - Nguyễn Văn Cường
        - Hồ Thị Quỳnh Như
        
        Giáo viên hướng dẫn: ThS. Khuất Thùy Phương
    """
)