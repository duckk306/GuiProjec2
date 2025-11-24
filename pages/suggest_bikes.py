import streamlit as st
import joblib
import pandas as pd

from utils.nlp_cleandata import clean_text_fromdf, clean_text
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

st.header("⭐ Gợi ý xe máy tương tự")
# ================== Feature lists ==================
num_cols = ['price', 'price_min', 'price_max', 'year_reg', 'km_driven', 'cc_numeric', 'price_segment_code', 'age']
flag_cols = ["is_moi", "is_do_xe", "is_su_dung_nhieu", "is_bao_duong", "is_do_ben", "is_phap_ly"]
cat_cols = ["brand", "vehicle_type", "model", "origin", "segment",'engine_size']

# Default brands fallback (in case user doesn't upload a dataset)
BRANDS = ['Aprilia','Bmw','Bazan','Benelli','Brixton','Cr&S','Daelim','Detech','Ducati','Gpx','Halim',
          'Harley Davidson','Honda','Hyosung','Hãng Khác','Ktm','Kawasaki','Keeway','Kengo','Kymco',
          'Moto Guzzi','Nioshima','Peugeot','Piaggio','Rebelusa','Royal Enfield','Sym','Sachs','Sanda',
          'Suzuki','Taya','Triumph','Vento','Victory','Vinfast','Visitor','Yamaha']

# ================== FUNCTIONS ==================     
@st.cache_resource 
def load_pipeline(suggest_model='model/cosine_sim_matrix.pkl',
                  tfidf='model/tfidf_vectorizer.pkl'):
    """
    Load cosine similarity matrix và vectorizer đã lưu trước đó.
    """
    cosine_matrix = joblib.load(suggest_model)
    vectorizer_loaded = joblib.load(tfidf)
    print("Đã load cosine matrix & vectorizer thành công.")
    return cosine_matrix, vectorizer_loaded

def recommend_by_id(data, item_id: int, top_n: int = 5):
    """
    Recommend similar motorbikes based on cosine similarity.
    Args:
        item_id (int): id hoặc index của xe trong DataFrame
        top_n (int): số lượng gợi ý muốn lấy
    Returns:
        DataFrame chứa các xe tương tự
    """
    if item_id not in data.index:
        raise ValueError(f"id {item_id} không tồn tại trong cơ sở của hệ thống")

    # Lấy hàng tương ứng trong ma trận cosine
    sim_scores = list(enumerate(cosine_sim_matrix[item_id]))

    # Sắp xếp theo độ tương đồng giảm dần, bỏ chính nó
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]

    # Lấy index xe tương tự
    similar_indices = [i[0] for i in sim_scores]
    similar_scores = [i[1] for i in sim_scores]

    # Tạo DataFrame kết quả
    recommendations = data.loc[similar_indices, ['id', 'Tiêu đề', 'Content']].copy()
    recommendations['similarity'] = similar_scores
    return recommendations.reset_index(drop=True)

def recommend_by_text(data, query: str, top_n: int = 5):
    """
    Gợi ý xe máy tương tự dựa trên văn bản người dùng nhập vào.
    
    Args:
        query (str): văn bản tìm kiếm
        top_n (int): số lượng gợi ý
    
    Returns:
        DataFrame: danh sách xe tương tự + độ tương đồng
    """

    # 1. Tiền xử lý query bằng hàm clean_text của bạn
    clean_query = clean_text(query)

    # 2. Vector hóa query
    query_vec = tfidf.transform([clean_query])

    # 3. Tính độ tương đồng cosine giữa query và toàn bộ item
    tfidf_matrix = tfidf.fit_transform(data['clean_text'])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 4. Lấy top N kết quả cao nhất
    top_idx = sims.argsort()[::-1][:top_n]
    top_scores = sims[top_idx]

    # 5. Trả về DataFrame kết quả
    result = data.iloc[top_idx][['id', 'Tiêu đề', 'Content']].copy()
    result["similarity"] = top_scores

    return result.reset_index(drop=True)


# ================== NAVIGATION ==================
st.sidebar.header("Navigation Menu")
st.sidebar.page_link("gui_project2.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/intro.py", label="Giới thiệu", icon="📃")
st.sidebar.page_link('pages/suggest_bikes.py', label='Gợi ý xe máy tương tự', icon="⭐")
st.sidebar.page_link('pages/predictprice_byclustering.py', label='Dự đoán giá xe máy cũ', icon="💵")
st.sidebar.page_link("pages/author.py", label="Thông tin tác giả", icon="ℹ️")


# ================== LOAD PIPELINE ==================
cosine_sim_matrix, tfidf = load_pipeline(suggest_model='model/cosine_sim_matrix.pkl',
                            tfidf='model/tfidf_vectorizer.pkl')
data_cleaned = pd.read_csv("data_motobikes_CleanForRecommendation.csv", sep=",")

# ================== RECOMMENDATION FROM FILE ==================
st.subheader("A. Chọn thông tin xe quan tâm từ file")

uploaded_file = st.file_uploader("Tải file Excel/CSV (data_motobikes.xlsx)", 
                                 type=["xlsx","csv"], key="pred_file")

if uploaded_file is not None:
    try:
        # 1. Read file
        if str(uploaded_file.name).lower().endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.success(f"Step 1. Đã đọc file: {uploaded_file.name} — {data.shape[0]} hàng")
        st.dataframe(data.head(5))
        
        # 2. Filter interested bikes
        #    2.1. Prepare Dropdown options
        price = st.number_input("Giá mong muốn (triệu VND)", min_value=0.0, value=10.0, step=0.1)
        price_min = st.number_input("Khoảng giá min (triệu VND)", min_value=0.0, value=8.0, step=0.1)
        price_max = st.number_input("Khoảng giá max (triệu VND)", min_value=0.0, value=12.0, step=0.1)
        
        last = st.session_state.get("last_clean")
        
        def options(feature, feature_list):
            opts = sorted(last[feature].dropna().unique().tolist()) if last is not None and feature in last.columns else feature_list
            return opts
        brands_opts = options("brand", BRANDS)
        models_opts = options("model", ["Wave","Exciter","Sirius"])
        vehicle_types_opts = options("vehicle_type", ["Xe số","Xe tay ga","Xe côn"])
        origin_opts = options("origin", ["Việt Nam","Nhập Khẩu"])
        segment_opts = options("segment", ["Phổ thông","Cận cao cấp","Cao cấp"])
        
        #     2.2. Prepare for inputs
        engine_size_sel = st.selectbox("Dung tích xe (nhãn)", options=["Dưới 50","50 - 100","100 - 175","Trên 175"], index=2)
        
        col1, col2 = st.columns(2)
        with col1:
            brand_inp = st.selectbox("Thương hiệu (brand)", options=brands_opts)
            model_inp = st.selectbox("Dòng xe (model)", options=models_opts)
            vehicle_type_inp = st.selectbox("Loại xe (vehicle_type)", options=vehicle_types_opts)
        with col2:
            km_driven = st.number_input("Số Km đã đi (km_driven)", min_value=0, step=1, value=1000)
            cc_numeric = st.number_input("Dung tích numeric (cc_numeric)", min_value=0, step=1, value=137)
            age = st.number_input("Tuổi xe (age)", min_value=0.1, step=0.1, value=3.0, format="%.1f")
        
        st.markdown("**Tình trạng (Tick = Có / Không = Không)**")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            is_moi = st.checkbox("is_moi", value=False)
        with r1c2:
            is_do_xe = st.checkbox("is_do_xe", value=False)
        with r1c3:
            is_su_dung_nhieu = st.checkbox("is_su_dung_nhieu", value=False)
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            is_bao_duong = st.checkbox("is_bao_duong", value=False)
        with r2c2:
            is_do_ben = st.checkbox("is_do_ben", value=False)
        with r2c3:
            is_phap_ly = st.checkbox("is_phap_ly", value=True)
            
        origin_inp = st.selectbox("Xuất xứ (origin)", options=origin_opts)
        segment_inp = st.selectbox("Phân khúc (segment)", options=segment_opts)
        segment_map = {"Phổ Thông": 1,
                        "Tầm Trung": 2,
                        "Cao Cấp": 3}
        price_segment_code = segment_map.get(segment_inp, 1) 
        
        #   2.3. Filter
        if st.button("🔍 Lọc xe quan tâm"):
            row = {
                "price": price,
                "price_min": price_min,
                "price_max": price_max,
                "km_driven": km_driven,
                "engine_size": engine_size_sel,
                "cc_numeric": cc_numeric,
                "age": age,
                "year_reg": 2025 - age,
                "price_segment_code": price_segment_code,
                "is_moi": int(is_moi),
                "is_do_xe": int(is_do_xe),
                "is_su_dung_nhieu": int(is_su_dung_nhieu),
                "is_bao_duong": int(is_bao_duong),
                "is_do_ben": int(is_do_ben),
                "is_phap_ly": int(is_phap_ly),
                "brand": brand_inp,
                "vehicle_type": vehicle_type_inp,
                "model": model_inp,
                "origin": origin_inp,
                "segment": segment_inp
            }
            
            df_filter = pd.DataFrame([row])
            st.dataframe(df_filter.head(10))
        

        # 3. Predict 
        if cosine_sim_matrix is None or tfidf is None:
            st.error("Model hoặc Vectorizer chưa được load (model_randomforest.pkl & tfidf_vectorizer.pkl).")
        else:
            try:
                sample_id = st.text_input("Nhập 1 ID xe được gợi ý ở trên")
                num_recommend = st.text_input("Số xe gợi ý được hiển thị", key="num_input_a")
                st.markdown("### Thông tin xe gốc:")

                st.text("Tiêu đề:" + data.loc[int(sample_id), "Tiêu đề"])
                st.text("Nội dung: " + data.loc[int(sample_id), "Mô tả chi tiết"])
                
                st.markdown("### Gợi ý các xe tương tự:")
                recommendation = recommend_by_id(data_cleaned, int(sample_id), top_n=int(num_recommend))
                recommendation_indicies = recommendation.index
                data_recomm = data[data.index.isin(recommendation_indicies)]
                
                overlapping_cols = recommendation.columns.intersection(data_recomm.columns)
                data_recomm_unique = data_recomm.drop(columns=overlapping_cols)
                merged_df = recommendation.join(data_recomm_unique, how='left')

                st.dataframe(merged_df)
                        
                
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
    except Exception as e:
        st.error(f"Lỗi khi đọc/tiền xử lý file: {e}")


# ================== RECOMMENDATION FROM TEXT ==================
st.subheader("B. Người dùng nhập mô tả xe quan tâm")
description = st.text_input("Nhập mô tả xe quan tâm")
st.markdown("*VD: xe còn mới, máy êm, hao xăng ít, đời từ 2019 trở lên. Nếu có Vision hoặc Janus chạy dưới 10.000km thì càng tốt.*")
num_recommend2 = st.text_input("Số xe gợi ý được hiển thị", key="num_input_b")

data = pd.read_excel("data_motobikes.xlsx")
recommendation2 = recommend_by_text(data_cleaned, description, top_n=int(num_recommend2))
recommendation_indicies2 = recommendation2.index
data_recomm2 = data[data.index.isin(recommendation_indicies2)] 
overlapping_cols = recommendation2.columns.intersection(data_recomm2.columns)
data_recomm_unique = data_recomm2.drop(columns=overlapping_cols)
merged_df2 = recommendation2.join(data_recomm_unique, how='left')

st.dataframe(merged_df2)

