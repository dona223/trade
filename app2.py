import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace
from pyvis.network import Network
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# 1. 스트림릿 페이지 기본 설정
st.set_page_config(page_title="K-EWS 공급망 안보 스트레스 테스트 시스템", layout="wide", page_icon="🔥")

# ====================================================
# [Backend] 데이터 로드 및 지식 그래프 매핑
# ====================================================
@st.cache_data
def load_final_kg_to_matrix(ttl_path, bok_coef, lithium_weight, bom_coef):
    g = Graph()
    g.parse(ttl_path, format="turtle")
    kews = Namespace("http://k-ews.org/ontology#")
    
    nodes = set()
    raw_edges = []
    for s, p, o in g:
        s_name = str(s).replace(str(kews), "")
        o_name = str(o).replace(str(kews), "")
        p_name = str(p).replace(str(kews), "")
        if str(s).startswith(str(kews)): nodes.add(s_name)
        if str(o).startswith(str(kews)): nodes.add(o_name)
        if str(s).startswith(str(kews)) and str(o).startswith(str(kews)):
            raw_edges.append((s_name, p_name, o_name))
            
    node_list = sorted(list(nodes))
    node_to_idx = {name: idx for idx, name in enumerate(node_list)}
    num_nodes = len(node_list)
    
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    excluded_nodes = set()
    for s, p, o in g:
        s_name = str(s).replace(str(kews), "")
        o_name = str(o).replace(str(kews), "")
        if "EXCLUDED" in str(p) or "EXCLUDED" in str(o):
            excluded_nodes.add(s_name)
            excluded_nodes.add(o_name)

    pure_lithium_coef = bok_coef * lithium_weight

    for s_name, p_name, o_name in raw_edges:
        if s_name in node_to_idx and o_name in node_to_idx:
            if s_name in excluded_nodes or o_name in excluded_nodes:
                continue
                
            s_idx = node_to_idx[s_name]
            o_idx = node_to_idx[o_name]
            
            if p_name == "PHYSICAL_INPUT_TO":
                if s_name == "Mat_Lithium" and o_name == "Mat_NCM_Cathode":
                    adj[s_idx, o_idx] = bom_coef
                elif s_name == "Mat_NCM_Cathode" and o_name == "Bat_Export_UnitPrice":
                    adj[s_idx, o_idx] = pure_lithium_coef
                else:
                    adj[s_idx, o_idx] = 0.150
            elif p_name == "LONG_TERM_CAUSES" or p_name == "DOMINO_EFFECT_TO":
                adj[s_idx, o_idx] = 0.75  
            elif p_name == "EXPLAINS_VARIANCE":
                adj[s_idx, o_idx] = 0.673 if s_name == "Mat_Lithium" else 0.219
            elif p_name == "TRIGGERS_SIGNAL":
                adj[s_idx, o_idx] = 0.50
            else:
                if adj[s_idx, o_idx] == 0:
                    adj[s_idx, o_idx] = 0.10  

    adj_with_self = adj + np.eye(num_nodes, dtype=np.float32)
    return torch.FloatTensor(adj_with_self), node_list, node_to_idx, excluded_nodes, raw_edges, adj

def generate_labels_v52(csv_path, target_date, node_to_idx):
    try:
        df = pd.read_csv(csv_path)
        df['YearMonth'] = pd.to_datetime(df['YearMonth'])
        target_dt = pd.to_datetime(target_date)
        row = df[(df['YearMonth'].dt.year == target_dt.year) & (df['YearMonth'].dt.month == target_dt.month)]
        if row.empty: return torch.randint(0, 3, (len(node_to_idx),)).long()
        row = row.iloc[0]
        stats = {col: (df[col].mean(), df[col].std()) for col in df.columns if col != 'YearMonth'}
        
        def get_status(col_name):
            if col_name not in row or pd.isna(row[col_name]): return 0
            val = row[col_name]
            mean, std = stats[col_name]
            if val >= mean + 2.0 * std: return 2
            elif val >= mean + 1.2 * std: return 1
            return 0

        labels = np.full(len(node_to_idx), -1, dtype=np.int64)
        mapping_rules = {
            "Mat_Lithium": "UP_탄산리튬", "Mat_Graphite_Other": "UP_흑연_기타", 
            "Bat_Export_UnitPrice": "Bat_Export_UnitPrice", "PPI_리튬이온": "PPI_리튬이온",
            "GPR_Channel": "GPR", "GSCPI_Channel": "GSCPI"
        }
        for node_name, idx in node_to_idx.items():
            if node_name in mapping_rules:
                labels[idx] = get_status(mapping_rules[node_name])
        return torch.LongTensor(labels)
    except:
        return torch.randint(0, 3, (len(node_to_idx),)).long()

# GNN 아키텍처 정의
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x, adj):
        return self.linear(torch.mm(adj, x))

class SupplyChainGNN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super(SupplyChainGNN, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, input_dim)
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
    def forward(self, adj):
        nodes_idx = torch.arange(adj.shape[0])
        x = self.node_embeddings(nodes_idx)
        x = F.relu(self.gcn1(x, adj))
        return self.gcn2(x, adj)


# ====================================================
# [Frontend] 대시보드 UI 및 인터랙티브 위젯 설계
# ====================================================
st.title("🔥 K-EWS 공급망 안보 GNN 스트레스 테스트 엔진")
st.markdown("한국은행 산업연관표 실측 데이터와 거시 경제 충격을 웹 슬라이더로 직접 조율하며 공급망 리스크 전파 파동을 시뮬레이션합니다.")
st.divider()

# 사이드바 컨트롤러 판넬
st.sidebar.header("⚙️ 1. 실증 통계 가중치 제어")
bok_coef = st.sidebar.slider("한국은행 전기장비 총투입계수", 0.01, 0.20, 0.085705, step=0.005, format="%.6f")
lithium_weight = st.sidebar.slider("비철금속 내 리튬 수입액 비중", 0.05, 0.50, 0.235211, step=0.005, format="%.6f")
bom_coef = st.sidebar.slider("양극재(중간재) 내 리튬 BOM 원가 비중", 0.10, 0.80, 0.450, step=0.01)

st.sidebar.header("⚡ 2. 수리적 캘리브레이션 밸브")
temperature = st.sidebar.slider("Temperature (확률 평활화 스펙트럼)", 1.0, 5.0, 2.5, step=0.1)
alpha_green = st.sidebar.slider("로짓 보정 (Green 하향 강도)", 0.5, 3.0, 1.5, step=0.1)
alpha_red = st.sidebar.slider("로짓 보정 (Red 상향 부스팅 강도)", 0.5, 3.0, 1.2, step=0.1)

st.sidebar.header("🚨 3. 가상 시나리오 쇼크 주입")
shock_gpr = st.sidebar.checkbox("지정학적 리스크 (GPR_Channel) 폭등", value=True)
shock_gscpi = st.sidebar.checkbox("글로벌 공급망 압력 (GSCPI_Channel) 폭등", value=True)
shock_graphite = st.sidebar.checkbox("흑연 수입 규제 (Mat_Graphite_Other) 공급 충격", value=True)

# 데이터 경로 바인딩
ttl_file_path = "K_EWS_KnowledgeGraph7.ttl" 
csv_file_path = "kews_preprocessed_132m_fixed.csv"

# 백엔드 엔진 가동 및 파라미터 맵핑
weighted_adj, node_list, node_to_idx, excluded_nodes, raw_edges, pure_adj_matrix = load_final_kg_to_matrix(
    ttl_file_path, bok_coef, lithium_weight, bom_coef
)
num_nodes = len(node_list)
dynamic_labels = generate_labels_v52(csv_file_path, "2023-10-01", node_to_idx)

# 경량 훈련 가동
model = SupplyChainGNN(num_nodes, 16, 8, 3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

model.train()
mask = dynamic_labels != -1
for epoch in range(1, 81):
    optimizer.zero_grad()
    output = model(weighted_adj)
    if mask.sum() > 0:
        loss = criterion(output[mask], dynamic_labels[mask])
        loss.backward()
        optimizer.step()

# 스트레스 테스트 추론 프로세스
model.eval()
with torch.no_grad():
    nodes_idx = torch.arange(num_nodes)
    emp_x = model.node_embeddings(nodes_idx)
    
    # 체크박스 가외 충격 바이어스 설계
    shock_mask = torch.zeros(num_nodes, 1)
    active_shocks = []
    if shock_gpr: active_shocks.append("GPR_Channel")
    if shock_gscpi: active_shocks.append("GSCPI_Channel")
    if shock_graphite: active_shocks.append("Mat_Graphite_Other")
    
    for s_node in active_shocks:
        if s_node in node_to_idx:
            s_idx = node_to_idx[s_node]
            emp_x[s_idx] += torch.tensor([5.0] * 16)
            shock_mask[s_idx] = 1.0
            
    x_stage1 = F.relu(model.gcn1(emp_x, weighted_adj))
    logits = model.gcn2(x_stage1, weighted_adj)
    
    # 실시간 로짓 밸런싱 연산
    for i in range(num_nodes):
        incoming_risk = torch.dot(weighted_adj[i], shock_mask.squeeze())
        if incoming_risk > 0:
            logits[i][0] -= incoming_risk * alpha_green
            logits[i][1] += incoming_risk * (alpha_red * 0.6)
            logits[i][2] += incoming_risk * alpha_red
            
    probabilities = F.softmax(logits / temperature, dim=1)

# --- 결과 시각화 레이아웃 드로잉 ---
st.subheader("📊 실증 데이터 기반 밸류체인 전파 결과 모니터링")

label_mapping = {
    "Mat_Anode_Active": "🔋 인조흑연 음극 활물질",
    "Mat_NCM_Cathode": "🧪 NCM 삼원계 양극재",
    "PPI_리튬이온": "📈 국내 배터리 생산자물가",
    "Bat_Export_UnitPrice": "🚢 배터리 수출단가",
    "Mat_RareEarth": "💎 희토류 (제외)"
}
status_emojis = {0: "🟢 안정", 1: "🟡 주의", 2: "🔴 위험"}

cols = st.columns(len(label_mapping))

for col, (eng_id, kor_name) in zip(cols, label_mapping.items()):
    with col:
        st.markdown(f"**{kor_name}**")
        st.caption(f"({eng_id})")
        
        if eng_id in node_to_idx:
            idx = node_to_idx[eng_id]
            
            if eng_id in excluded_nodes:
                prob_green, prob_yellow, prob_red = 100.0, 0.0, 0.0
                max_class = 0
                st.info("⚠️ 통계적 탈락 품목\n(리스크 전파 차단)")
            else:
                prob_green  = probabilities[idx][0].item() * 100
                prob_yellow = probabilities[idx][1].item() * 100
                prob_red    = probabilities[idx][2].item() * 100
                max_class = torch.argmax(probabilities[idx]).item()
            
            if max_class == 0:   st.success(f"최종 판정: {status_emojis[max_class]}")
            elif max_class == 1: st.warning(f"최종 판정: {status_emojis[max_class]}")
            else:                st.error(f"최종 판정: {status_emojis[max_class]}")
            
            st.write(f"🟢 Green (안정): {prob_green:.1f}%")
            st.write(f"🟡 Yellow (주의): {prob_yellow:.1f}%")
            st.write(f"🔴 Red (위험): {prob_red:.1f}%")
            st.progress(int(prob_red))

# ====================================================
# 💡 [핵심 추가] 인터랙티브 리스크 흐름 유향 그래프 연동
# ====================================================
st.divider()
st.subheader("🎯 실시간 리스크 인과 파동 전파 맵 (PyVis Interactive Network)")
st.markdown("마우스 휠로 **확대/축소**가 가능하며, 노드를 **드래그**하여 구조를 자유롭게 펼칠 수 있습니다. 쇼크 주입 시 에지 두께와 색상이 변합니다.")

# PyVis 네트워크 객체 생성 (유향 그래프 지정)
net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=True)

# 1. 노드 시각화 스펙 설계
for node_name, idx in node_to_idx.items():
    # 라벨 다듬기 (맵핑된 이름이 있으면 표기, 없으면 원본 ID 활용)
    clean_label = node_name
    for eng_k, kor_v in label_mapping.items():
        if node_name == eng_k:
            clean_label = kor_v.split(" ")[1] # 이모지 뒤 한글명만 추출
            
    # 노드 상태별 색상 및 크기 설정 지표 정의
    if node_name in active_shocks:
        node_color = "#ff4b4b" # 강렬한 빨간색 (충격 근원지)
        node_size = 35
        node_label = f"🔥 {clean_label}\n[쇼크 진원지]"
    elif node_name in excluded_nodes:
        node_color = "#b0bec5" # 회색 (차단 노드)
        node_size = 15
        node_label = f"⚪ {clean_label}\n[리스크 차단]"
    else:
        # GNN 예측결과 위험도(Red 비중)에 따라 동적 칼러 스케일링
        red_ratio = probabilities[idx][2].item()
        yellow_ratio = probabilities[idx][1].item()
        
        if red_ratio > 0.4:
            node_color = "#ff8a80" # 주황빛 도는 빨강
            node_size = 28
        elif yellow_ratio > 0.4:
            node_color = "#ffe082" # 연노랑
            node_size = 23
        else:
            node_color = "#a5d6a7" # 연초록 (안정)
            node_size = 20
        node_label = clean_label

    net.add_node(node_name, label=node_label, color=node_color, size=node_size, shape="dot")

# 2. 동적 가중치 에지(선) 시각화 적용
for s_name, p_name, o_name in raw_edges:
    if s_name in node_to_idx and o_name in node_to_idx:
        if s_name in excluded_nodes or o_name in excluded_nodes:
            continue
            
        s_idx = node_to_idx[s_name]
        o_idx = node_to_idx[o_name]
        
        # 실제 연산에 활용된 순방향 유효 가중치 강도 추출
        edge_weight = pure_adj_matrix[s_idx, o_idx]
        
        # 현재 출발지 노드가 충격을 받은 상태라면 연결선 강조 (빨간색 전파 흐름 효과)
        if s_name in active_shocks or (s_name == "Mat_Lithium" and "GPR_Channel" in active_shocks):
            edge_color = "#e53935" # 리스크 전파중인 활성화 라인 (진한 빨강)
            edge_width = float(edge_weight * 8.0) # 선 두께 대폭 스케일업
        else:
            edge_color = "#cfd8dc" # 평시 흐름 라인 (은은한 회색)
            edge_width = max(float(edge_weight * 3.0), 1.0)
            
        edge_title = f"관계성: {p_name}\n유효 전파 가중치: {edge_weight:.4f}"
        net.add_edge(s_name, o_name, value=edge_width, color=edge_color, title=edge_title)

# 물리 엔진 설정 최적화 (노드들이 꼬이지 않고 이쁘게 정렬되도록 세팅)
net.toggle_physics(True)
net.set_options("""
var options = {
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -4000,
      "centralGravity": 0.2,
      "springLength": 120,
      "springConstant": 0.04
    },
    "minVelocity": 0.75
  }
}
""")

# HTML 파일로 빌드 후 스트림릿 컴포넌트로 화면에 주입
try:
    net.save_graph("pyvis_graph.html")
    HtmlFile = open("pyvis_graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=520)
except Exception as e:
    st.error(f"그래프 렌더링 중 일시적 입출력 지연이 발생했습니다. 잠시 후 새로고침해 주세요. ({e})")

st.divider()
st.subheader("💡 실시간 시뮬레이션 인사이트 브리핑")
st.markdown(f"""
* **현재 활성화된 쇼크 파동**: {', '.join(active_shocks) if active_shocks else '⚠️ 없음 (평시 안보 모드)'}
* **실시간 계산된 연쇄 유효 가중치**: 현재 설정한 가중치 조합에 의거하여, 리튬 가격 변동이 최종 배터리 가격에 미치는 순수 산업연관 영향력은 에지당 **{(bok_coef * lithium_weight * 100):.4f}%** 강도로 수렴하여 신경망 연산에 주입되고 있습니다.
* **시뮬레이터 반응 유기성 안내**: 왼쪽 사이드바에서 거시 충격 원인을 변경하면 **상단 라벨 카드들의 확률 수치**뿐만 아니라, **하단 그래픽 맵의 에지 색상과 노드 크기**가 실시간 인과 전파 경로(Causal Path)에 맞춰 완전히 인터랙티브하게 재구조화됩니다.
""")