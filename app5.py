from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable
)
from reportlab.platypus import PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import os
import time

import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
import numpy as np
import pandas as pd
# [수정] URIRef 임포트 추가
from rdflib import Graph, Namespace, RDF, URIRef
from pyvis.network import Network
import streamlit.components.v1 as components

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="K-EWS 공급망 이벤트 영향 분석 시스템",
    layout="wide", page_icon="🔬"
)

# ══════════════════════════════════════════════
# [Config] Gemini API 키 입력 (사이드바 최상단)
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("🤖 Gemini AI 브리핑 설정")
    gemini_api_key = st.text_input(
        "Gemini API Key", type="password",
        placeholder="AIza..."
    )
    gemini_model = st.selectbox(
        "모델 선택",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0
    )


# ══════════════════════════════════════════════
# [Backend 0] Gemini API 브리핑 생성 함수
# ══════════════════════════════════════════════
def generate_gemini_briefing(
    api_key: str,
    model: str,
    active_shocks: list,
    result_summary: dict,  # {node_name: {status, pg, py, pr}}
    bok_coef: float,
    lithium_weight: float,
    tw_highlights: list,   # [(s, o, tw_val), ...] 상위 Tw 엣지
) -> str:
    if not api_key:
        return None

    # 결과 텍스트 구성
    result_lines = []
    for node, info in result_summary.items():
        result_lines.append(
            f"- {node}: {info['status']} "
            f"(영향 미미 {info['pg']:.1f}% / 주의 {info['py']:.1f}% / "
            f"심각 {info['pr']:.1f}%)"
        )
    tw_lines = [
        f"- {s} ↔ {o}: Tw={tw:.4f}" for s, o, tw in tw_highlights[:5]
    ]

    prompt = f"""당신은 한국 공급망 안보 전문 애널리스트입니다.
아래 K-EWS 시스템의 GNN 분석 결과와 **당신이 구글 검색을 통해 파악한 최신 한국 무역/원자재 통계 정보**를 융합하여 한국 배터리 공급망에 미치는 이벤트 파급 영향을 전문적이고 간결하게 브리핑해 주세요.

## 현재 발생한 이벤트 (쇼크 시나리오)
{', '.join(active_shocks) if active_shocks else '없음 (기저 상태)'}

## GNN 파급 영향 예측 결과
{chr(10).join(result_lines)}

## NetCrafter Tw 기준 상위 의미론적 연결 경로
{chr(10).join(tw_lines) if tw_lines else '없음'}

## 실증 가중치 설정값
- 한국은행 전기장비 총투입계수: {bok_coef:.6f}
- 리튬 수입액 비중: {lithium_weight:.6f}
- 산업연관 유효 가중치: {bok_coef * lithium_weight * 100:.4f}%

## 브리핑 요구사항
1. 이벤트 성격과 파급 메커니즘 (2~3문장)
2. 가장 위험한 원자재/지표와 그 이유 (2문장)
3. **구글 검색을 통해 파악한 최근 1~2년 내 실제 한국의 총 수입액 대비 탄산리튬/흑연의 대략적인 무역 규모 비중이나 최신 공급망 동향 언급**
4. 정책/기업 대응 시사점 (2문장)
5. 불확실성 및 주의 사항 (1문장)

한국어로, 600자 내외로 작성하세요."""

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    # [수정] 구글 검색 증강(Grounding) 툴 활성화 및 모델별 호환성 보정
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096} # 통계 검색 시 정확도를 위해 temperature 하향
    }
    
    # gemini-2.0-flash 혹은 gemini-1.5-pro 등의 텍스트 모델에서 검색 기능 활성화
    payload["tools"] = [{"googleSearch": {}}]
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        candidate = resp.json()["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]

        # 잘렸는지 확인
        finish_reason = candidate.get("finishReason", "")
        if finish_reason == "MAX_TOKENS":
            text += "\n\n⚠️ (응답이 토큰 제한으로 중간에 잘렸습니다. maxOutputTokens를 늘려주세요.)"

        return text
    except Exception as e:
        return f"⚠️ Gemini API 오류: {e}"
# ══════════════════════════════════════════════
# [Backend 0-1] Gemini Grounding 기반 실시간 통계 수집 엔진 (신규 추가)
# ══════════════════════════════════════════════
def fetch_real_trade_weight(api_key: str, model: str) -> float:
    """Gemini 구글 검색 증강 기능을 이용해 최신 수입 비중 데이터를 파싱하여 리턴합니다."""
    if not api_key:
        return 0.05 # API 키 누락 시 백업용 기본 가중치 적용

    prompt = """최근 1개년 기준 한국의 총 수입액 대비 탄산리튬 수입액 비중을 검색해서 계산해줘. 
다른 설명이나 텍스트는 일체 하지 말고, 오직 소수점 수치(예: 0.0143) 딱 하나만 텍스트로 대답해라. 
만약 수치를 특정하기 어렵다면 무조건 0.05 라고만 출력해."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10},
        "tools": [{"googleSearch": {}}]
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 429:
                wait = 2 ** attempt  # 1, 2, 4초 백오프
                time.sleep(wait)
                continue
            resp.raise_for_status()
            ans = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            return float(ans)
        except Exception:
            if attempt == max_retries - 1:
                return 0.05
            time.sleep(1)
    return 0.05
# ══════════════════════════════════════════════
# [Backend 1] NetCrafter Tw 계산 엔진 (수정본)
# ══════════════════════════════════════════════
def calculate_ontology_tw(g, KEWS, trade_weight=0.05): # [수정] 외부에서 가중치를 주입받도록 인자 추가 (기본값 0.05)
    all_materials = list(set(
        str(s) for s in g.subjects()
        if isinstance(s, URIRef) and str(s).startswith(str(KEWS))
    ))
    total_N = len(all_materials)
    if total_N == 0:
        return {}, {}, []

    target_set = set()
    for s, p, o in g.triples((None, None, None)):
        if any(k in str(o) for k in ["Alert_Red", "Alert_Yellow", "Shock"]):
            if str(s) in all_materials:
                target_set.add(str(s))
    if not target_set:
        target_set = set(all_materials[:max(1, int(total_N * 0.3))])
    target_M = len(target_set)

    weights = {}
    feature_nodes = set(
        str(o) for _, _, o in g
        if isinstance(o, URIRef) and str(o).startswith(str(KEWS))
    )
    
    # [수정] 하드코딩 0.05를 지우고 Gemini 연동을 위해 외부 주입 변수 매핑
    trade_volume_to_gdp = trade_weight 

    for fn in feature_nodes:
        if any(fn.endswith(t) for t in
               ["RawMaterial", "Framework", "MainEngine", "CoreMaterial", "StatisticalRelation"]):
            continue
        
        assoc = set(str(s) for s, _, _ in g.triples((None, None, URIRef(fn)))) & set(all_materials)
        
        a = len(assoc & target_set)
        b = len(assoc) - a
        c = target_M - a
        d = (total_N - target_M) - b
        
        _, p_val = stats.fisher_exact([[a, b], [c, d]])
        
        weights[fn] = -np.log10(max(p_val, 1e-10)) * trade_volume_to_gdp

    mat_feats = {m: set() for m in all_materials}
    for m in all_materials:
        for _, _, o in g.triples((URIRef(m), None, None)):
            if isinstance(o, URIRef) and str(o) in weights:
                mat_feats[m].add(str(o))

    tw_map = {}
    for i in range(total_N):
        for j in range(i, total_N):
            m1, m2 = all_materials[i], all_materials[j]
            inter = mat_feats[m1] & mat_feats[m2]
            union = mat_feats[m1] | mat_feats[m2]
            denom = sum(weights[f] for f in union)
            tw = sum(weights[f] for f in inter) / denom if denom > 0 else 0.0
            tw_map[(m1, m2)] = tw_map[(m2, m1)] = tw

    return weights, tw_map, all_materials

# ══════════════════════════════════════════════
# [Backend 2] KG 파싱 + BoK × Tw 융합
# ══════════════════════════════════════════════
@st.cache_data
def load_kg_with_tw_fusion(ttl_path, bok_coef, lithium_weight, bom_coef, trade_weight=0.05): # [수정] trade_weight 인자 추가
    g = Graph()
    g.parse(ttl_path, format="turtle")
    kews = Namespace("http://k-ews.org/ontology#")

    # [수정] 계산 엔진 호출 시 동적 무역 비중 가중치 전달
    _, tw_map, all_material_uris = calculate_ontology_tw(g, kews, trade_weight=trade_weight)
    uri_to_short = {uri: uri.replace(str(kews), "") for uri in all_material_uris}
    short_to_uri = {v: k for k, v in uri_to_short.items()}

    nodes, raw_edges = set(), []
    for s, p, o in g:
        s_n = str(s).replace(str(kews), "")
        o_n = str(o).replace(str(kews), "")
        p_n = str(p).replace(str(kews), "")
        if str(s).startswith(str(kews)): nodes.add(s_n)
        if str(o).startswith(str(kews)): nodes.add(o_n)
        if str(s).startswith(str(kews)) and str(o).startswith(str(kews)):
            raw_edges.append((s_n, p_n, o_n))

    node_list = sorted(nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    excluded_nodes = set()
    for s, p, o in g:
        if "EXCLUDED" in str(p) or "EXCLUDED" in str(o):
            excluded_nodes.update([
                str(s).replace(str(kews), ""),
                str(o).replace(str(kews), "")
            ])

    pure_lithium_coef = bok_coef * lithium_weight
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for s_n, p_n, o_n in raw_edges:
        if s_n in excluded_nodes or o_n in excluded_nodes: continue
        if s_n not in node_to_idx or o_n not in node_to_idx: continue
        si, oi = node_to_idx[s_n], node_to_idx[o_n]

        if p_n == "PHYSICAL_INPUT_TO":
            if s_n == "Mat_Lithium" and o_n == "Mat_NCM_Cathode":
                base_w = bom_coef
            elif s_n == "Mat_NCM_Cathode" and o_n == "Bat_Export_UnitPrice":
                base_w = pure_lithium_coef
            else:
                base_w = 0.150
        elif p_n in ("LONG_TERM_CAUSES", "DOMINO_EFFECT_TO"):
            base_w = 0.75
        elif p_n == "EXPLAINS_VARIANCE":
            base_w = 0.673 if s_n == "Mat_Lithium" else 0.219
        elif p_n == "TRIGGERS_SIGNAL":
            base_w = 0.50
        else:
            base_w = 0.10

        su = short_to_uri.get(s_n)
        ou = short_to_uri.get(o_n)
        tw_val = tw_map.get((su, ou), 0.0) if (su and ou) else 0.0
        adj[si, oi] = base_w * (1.0 + tw_val)

    adj_with_self = adj + np.eye(num_nodes, dtype=np.float32)
    return (
        torch.FloatTensor(adj_with_self),
        node_list, node_to_idx, excluded_nodes,
        raw_edges, adj, tw_map, short_to_uri
    )


# ══════════════════════════════════════════════
# [Backend 3] 시계열 레이블 생성
# ══════════════════════════════════════════════
def generate_labels(csv_path, target_date, node_to_idx):
    try:
        df = pd.read_csv(csv_path)
        df['YearMonth'] = pd.to_datetime(df['YearMonth'])
        target_dt = pd.to_datetime(target_date)
        row = df[
            (df['YearMonth'].dt.year == target_dt.year) &
            (df['YearMonth'].dt.month == target_dt.month)
        ]
        if row.empty:
            return torch.randint(0, 3, (len(node_to_idx),)).long()
        row = row.iloc[0]
        stat_map = {c: (df[c].mean(), df[c].std())
                    for c in df.columns if c != 'YearMonth'}
        def get_status(col):
            if col not in row or pd.isna(row[col]): return 0
            v, (m, s) = row[col], stat_map[col]
            if v >= m + 2.0 * s: return 2
            elif v >= m + 1.2 * s: return 1
            return 0
        labels = np.full(len(node_to_idx), -1, dtype=np.int64)
        mapping = {
            "Mat_Lithium":       "UP_탄산리튬",
            "Mat_Graphite_Other":"UP_흑연_기타",
            "Bat_Export_UnitPrice":"Bat_Export_UnitPrice",
            "PPI_리튬이온":       "PPI_리튬이온",
            "GPR_Channel":       "GPR",
            "GSCPI_Channel":     "GSCPI",
        }
        for node, idx in node_to_idx.items():
            if node in mapping:
                labels[idx] = get_status(mapping[node])
        return torch.LongTensor(labels)
    except:
        return torch.randint(0, 3, (len(node_to_idx),)).long()


# ══════════════════════════════════════════════
# [Backend 4] GNN 아키텍처
# ══════════════════════════════════════════════
class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)
    def forward(self, x, adj):
        return self.linear(torch.mm(adj, x))

class SupplyChainGNN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.emb  = nn.Embedding(num_nodes, input_dim)
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
    def forward(self, adj):
        x = self.emb(torch.arange(adj.shape[0]))
        x = F.relu(self.gcn1(x, adj))
        return self.gcn2(x, adj)


# ══════════════════════════════════════════════
# [Backend 5] PDF 보고서 생성
# ══════════════════════════════════════════════
def generate_pdf_report(
    active_shocks, result_summary, briefing_text,
    bok_coef, lithium_weight, bom_coef, tw_highlights
):
    buffer = BytesIO()

    # ── 한글 폰트 등록 (윈도우 기본 폰트 우선, 없으면 폴백) ──
    font_name = "Helvetica"  # 기본 폴백
    font_candidates = [
        ("C:/Windows/Fonts/malgun.ttf",    "MalgunGothic"),
        ("C:/Windows/Fonts/NanumGothic.ttf", "NanumGothic"),
        ("C:/Windows/Fonts/gulim.ttc",     "Gulim"),
    ]
    for font_path, name in font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(name, font_path))
                font_name = name
                break
            except Exception:
                continue

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm
    )

    # ── 스타일 정의 ──
    styles = getSampleStyleSheet()
    def make_style(name, parent="Normal", **kwargs):
        return ParagraphStyle(name, parent=styles[parent],
                              fontName=font_name, **kwargs)

    s_title    = make_style("Title2",   fontSize=16, textColor=colors.HexColor("#1a252f"),
                             spaceAfter=6, leading=22)
    s_h2       = make_style("H2",       fontSize=12, textColor=colors.HexColor("#2980b9"),
                             spaceBefore=14, spaceAfter=4, leading=16)
    s_body     = make_style("Body",     fontSize=9,  leading=14, spaceAfter=4)
    s_meta     = make_style("Meta",     fontSize=8,  textColor=colors.HexColor("#555555"),
                             leading=12)
    s_briefing = make_style("Brief",    fontSize=9,  leading=16,
                             leftIndent=8, rightIndent=8)
    s_footer   = make_style("Footer",   fontSize=7,  textColor=colors.HexColor("#888888"),
                             alignment=1)

    story = []

    # ── 제목 ──
    story.append(Paragraph("K-EWS 공급망 이벤트 파급 영향 분석 보고서", s_title))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#2980b9")))
    story.append(Spacer(1, 4*mm))

    # ── 메타 정보 ──
    shock_str = ', '.join(active_shocks) if active_shocks else '없음 (기저 상태)'
    now_str   = pd.Timestamp.now().strftime('%Y년 %m월 %d일 %H:%M')
    story.append(Paragraph(f"분석 시점: {now_str}  |  이벤트 시나리오: {shock_str}", s_meta))
    story.append(Paragraph(
        f"BoK 총투입계수: {bok_coef:.6f}  |  "
        f"리튬 수입 비중: {lithium_weight:.6f}  |  "
        f"BOM 비중: {bom_coef:.3f}  |  "
        f"유효 가중치: {bok_coef * lithium_weight * 100:.4f}%",
        s_meta
    ))
    story.append(Spacer(1, 4*mm))

    # ── ① 파급 영향 예측 결과 테이블 ──
    story.append(Paragraph("① GNN 파급 영향 예측 결과", s_h2))

    status_color_map = {
        "🟢 영향 미미":   colors.HexColor("#27ae60"),
        "🟡 주의 필요":   colors.HexColor("#f39c12"),
        "🔴 심각한 영향": colors.HexColor("#e74c3c"),
        "분석 제외":       colors.HexColor("#95a5a6"),
    }

    tbl_data = [["원자재 / 지표", "파급 판정", "영향 미미", "주의 필요", "심각한 영향"]]
    row_colors = []
    for i, (node, info) in enumerate(result_summary.items()):
        tbl_data.append([
            node,
            info['status'].replace("🟢","").replace("🟡","").replace("🔴","").strip(),
            f"{info['pg']:.1f}%",
            f"{info['py']:.1f}%",
            f"{info['pr']:.1f}%",
        ])
        row_colors.append(status_color_map.get(info['status'], colors.HexColor("#95a5a6")))

    tbl = Table(tbl_data, colWidths=[52*mm, 30*mm, 24*mm, 24*mm, 30*mm])
    tbl_style = TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), font_name),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#2980b9")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ALIGN",       (0,0), (0,-1),  "LEFT"),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ])
    # 판정 컬럼 행별 색상 적용
    for i, c in enumerate(row_colors):
        tbl_style.add("BACKGROUND", (1, i+1), (1, i+1), c)
        tbl_style.add("TEXTCOLOR",  (1, i+1), (1, i+1), colors.white)
    tbl.setStyle(tbl_style)
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

    # ── ② Tw 상위 경로 테이블 ──
    story.append(Paragraph("② NetCrafter Tw 상위 의미론적 연결 경로", s_h2))
    if tw_highlights:
        tw_data = [["출발 노드", "도착 노드", "Tw 유사도"]]
        for s_uri, o_uri, tw_val in tw_highlights[:8]:
            s_short = str(s_uri).split("#")[-1].split("/")[-1]
            o_short = str(o_uri).split("#")[-1].split("/")[-1]
            tw_data.append([s_short, o_short, f"{tw_val:.4f}"])
        tw_tbl = Table(tw_data, colWidths=[70*mm, 70*mm, 30*mm])
        tw_tbl.setStyle(TableStyle([
            ("FONTNAME",   (0,0), (-1,-1), font_name),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("BACKGROUND", (0,0), (-1,0),  colors.HexColor("#2980b9")),
            ("TEXTCOLOR",  (0,0), (-1,0),  colors.white),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#f8f9fa"), colors.white]),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(tw_tbl)
    else:
        story.append(Paragraph("원자재 간 Tw 데이터가 없습니다.", s_body))
    story.append(Spacer(1, 4*mm))

    # ── ③ AI 브리핑 ──
    story.append(Paragraph(f"③ AI 파급 영향 브리핑 (Gemini)", s_h2))
    brief = briefing_text or "Gemini API가 연동되지 않은 상태입니다."
    # 줄바꿈 → ReportLab <br/> 변환
    for line in brief.split("\n"):
        line = line.strip()
        if line:
            story.append(Paragraph(line, s_briefing))
    story.append(Spacer(1, 6*mm))

    # ── 푸터 ──
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#bdc3c7")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "본 보고서는 K-EWS 시스템이 자동 생성한 분석 결과입니다. "
        "실제 정책 결정 시 전문가 검토를 병행하시기 바랍니다.",
        s_footer
    ))

    doc.build(story)
    return buffer.getvalue()


# ══════════════════════════════════════════════
# [Frontend] 타이틀
# ══════════════════════════════════════════════
st.title("🔬 K-EWS 공급망 이벤트 파급 영향 분석 시스템")
st.markdown(
    "**이벤트 발생 시 핵심 원자재·지표에 미치는 파급 경로**를 "
    "GNN + NetCrafter $T_w$ 온톨로지 유사도로 정량 분석합니다."
)
st.divider()

with st.expander("📘 시뮬레이터 조절 변수 가이드라인", expanded=False):
    st.markdown("""
#### ⚙️ 1. 실증 통계 가중치 제어
- **한국은행 전기장비 총투입계수** (`기준: 0.085705`): 이차전지 산업 실측 상호의존도
- **비철금속 내 리튬 수입액 비중** (`기준: 0.235211`): 리튬 무역 비중
- **양극재 내 리튬 BOM 원가 비중** (`기준: 0.45`): 중간재 원가 구성비

#### 🧬 2. NetCrafter Tw 융합 원리
$$\\text{final\\_weight} = w_{\\text{BoK}} \\times (1 + T_w)$$
Fisher's Exact Test 기반 온톨로지 유사도가 BoK 가중치를 의미론적으로 보정합니다.

#### ⚡ 3. 캘리브레이션 밸브
- **Temperature**: 확률 분포 평활화 (기본값 `2.5`)
- **Green 하향 강도**: '영향 없음' 과잉 판정 억제
- **Red 상향 부스팅**: '심각한 영향' 민감도 증폭
    """)

# ── 사이드바 컨트롤러 ──
st.sidebar.header("⚙️ 1. 실증 통계 가중치 제어")
bok_coef       = st.sidebar.slider("한국은행 전기장비 총투입계수", 0.01, 0.20, 0.085705, step=0.005, format="%.6f")
lithium_weight = st.sidebar.slider("비철금속 내 리튬 수입액 비중", 0.05, 0.50, 0.235211, step=0.005, format="%.6f")
bom_coef       = st.sidebar.slider("양극재(중간재) 내 리튬 BOM 원가 비중", 0.10, 0.80, 0.450, step=0.01)

st.sidebar.header("⚡ 2. 수리적 캘리브레이션 밸브")
temperature = st.sidebar.slider("Temperature", 1.0, 5.0, 2.5, step=0.1)
alpha_green = st.sidebar.slider("로짓 보정 (Green 하향 강도)", 0.5, 3.0, 1.5, step=0.1)
alpha_red   = st.sidebar.slider("로짓 보정 (Red 상향 부스팅)", 0.5, 3.0, 1.2, step=0.1)

st.sidebar.header("🚨 3. 이벤트 시나리오 주입")
shock_gpr      = st.sidebar.checkbox("지정학적 리스크 (GPR_Channel) 폭등",             value=True)
shock_gscpi    = st.sidebar.checkbox("글로벌 공급망 압력 (GSCPI_Channel) 폭등",        value=True)
shock_graphite = st.sidebar.checkbox("흑연 수입 규제 (Mat_Graphite_Other) 공급 충격", value=True)
shock_covid    = st.sidebar.checkbox("팬데믹급 공급망 붕괴 (Event_COVID19) 재현",      value=False)

# ── 네트워크 시각화 옵션 ──
st.sidebar.header("🗺️ 4. 네트워크 맵 표시 설정")
edge_scale_mode = st.sidebar.radio(
    "엣지 두께 스케일링 방식",
    ["로그 스케일 (분산 균형)", "선형 스케일 (원본)"],
    index=0
)
show_weak_edges = st.sidebar.checkbox("약한 엣지 표시 (임계값 이하)", value=False)
edge_threshold  = st.sidebar.slider("엣지 표시 임계값 (가중치)", 0.0, 1.0, 0.05, step=0.01)
max_edge_width  = st.sidebar.slider("최대 엣지 두께 상한", 3.0, 20.0, 8.0, step=1.0)


# ══════════════════════════════════════════════
# [Engine] 백엔드 실행 (GNN 모델 연산 유실 복구 및 최적화 버전)
# ══════════════════════════════════════════════
TTL_PATH = "K_EWS_KnowledgeGraph7.ttl"
CSV_PATH = "kews_preprocessed_132m_fixed.csv"

# 1 세션 상태 초기화
if "dynamic_trade_weight" not in st.session_state:
    st.session_state.dynamic_trade_weight = 0.05
if "trade_weight_fetched" not in st.session_state:
    st.session_state.trade_weight_fetched = False  # 성공적으로 채굴했는지 여부

# 2 Gemini Grounding으로 실시간 가중치 채굴 — "아직 시도 안 했을 때만" 1회 호출
if gemini_api_key and not st.session_state.trade_weight_fetched:
    with st.spinner("🔍 Gemini가 관세청/구글 통계에서 리튬 실제 무역 비중 가중치를 실시간 채굴 중..."):
        mined_w = fetch_real_trade_weight(gemini_api_key, gemini_model)
        st.session_state.dynamic_trade_weight = mined_w
        st.session_state.trade_weight_fetched = True  # 성공/실패 무관하게 재시도 차단

# 사이드바 화면 적절한 곳에 마이닝된 수치 실시간 시각화
st.sidebar.info(f"🧬 Gemini 동적 마이닝 가중치: {st.session_state.dynamic_trade_weight:.5f}")

# 3. [KG 파싱 & Tw 유사도 융합] 채굴된 동적 가중치(trade_weight)를 엔진에 주입
(weighted_adj, node_list, node_to_idx,
 excluded_nodes, raw_edges, pure_adj,
 tw_map, short_to_uri) = load_kg_with_tw_fusion(
    TTL_PATH, bok_coef, lithium_weight, bom_coef,
    trade_weight=st.session_state.dynamic_trade_weight  # 수치 동적 반영
)

num_nodes      = len(node_list)
dynamic_labels = generate_labels(CSV_PATH, "2023-10-01", node_to_idx)

# 4. [GNN 인공지능 학습] 임베딩 레이어 및 GCN 모델 파이프라인 가동 (수정본에서 유실되었던 코드 복구)
model = SupplyChainGNN(num_nodes, 16, 8, 3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

model.train()
mask = dynamic_labels != -1
for _ in range(80):
    optimizer.zero_grad()
    out = model(weighted_adj)
    if mask.sum() > 0:
        loss = criterion(out[mask], dynamic_labels[mask])
        loss.backward()
        optimizer.step()

# 5. [쇼크 주입 & 추론] 시나리오 기반 파급 효과 시뮬레이션 가동
model.eval()
with torch.no_grad():
    emp_x = model.emb(torch.arange(num_nodes))

    active_shocks = []
    if shock_gpr:      active_shocks.append("GPR_Channel")
    if shock_gscpi:    active_shocks.append("GSCPI_Channel")
    if shock_graphite: active_shocks.append("Mat_Graphite_Other")
    if shock_covid:    active_shocks.append("Event_COVID19")

    shock_mask = torch.zeros(num_nodes, 1)
    for s in active_shocks:
        if s in node_to_idx:
            idx = node_to_idx[s]
            emp_x[idx] += torch.tensor([5.0] * 16)
            shock_mask[idx] = 1.0

    x1     = F.relu(model.gcn1(emp_x, weighted_adj))
    logits = model.gcn2(x1, weighted_adj)

    # 6. [캘리브레이션 밸브] 로짓 변환 및 보정 가중치 연산
    for i in range(num_nodes):
        incoming = torch.dot(weighted_adj[i], shock_mask.squeeze())
        if incoming > 0:
            logits[i][0] -= incoming * alpha_green
            logits[i][1] += incoming * (alpha_red * 0.6)
            logits[i][2] += incoming * alpha_red

    # 최종 결과물 확률 도출 (하단 결과 카드 및 네트워크 매핑 바인딩용)
    probabilities = F.softmax(logits / temperature, dim=1)

# ══════════════════════════════════════════════
# [Frontend] 결과 카드
# ══════════════════════════════════════════════
st.subheader("📊 이벤트 파급 영향 분석 결과")

label_mapping = {
    "Mat_Anode_Active":    "🔋 인조흑연 음극 활물질",
    "Mat_NCM_Cathode":     "🧪 NCM 삼원계 양극재",
    "PPI_리튬이온":         "📈 국내 배터리 생산자물가",
    "Bat_Export_UnitPrice":"🚢 배터리 수출단가",
    "Mat_RareEarth":       "💎 희토류 (분석 제외)",
}
status_emojis = {0: "🟢 영향 미미", 1: "🟡 주의 필요", 2: "🔴 심각한 영향"}

result_summary = {}  # PDF·Gemini 전달용
cols = st.columns(len(label_mapping))
for col, (eng_id, kor_name) in zip(cols, label_mapping.items()):
    with col:
        st.markdown(f"**{kor_name}**")
        st.caption(f"({eng_id})")
        if eng_id not in node_to_idx:
            continue
        idx = node_to_idx[eng_id]
        if eng_id in excluded_nodes:
            st.info("⚠️ 분석 제외 품목")
            result_summary[kor_name] = {"status":"분석 제외","pg":100,"py":0,"pr":0}
        else:
            pg = probabilities[idx][0].item() * 100
            py = probabilities[idx][1].item() * 100
            pr = probabilities[idx][2].item() * 100
            mc = torch.argmax(probabilities[idx]).item()
            result_summary[kor_name] = {"status": status_emojis[mc], "pg": pg, "py": py, "pr": pr}
            if mc == 0: st.success(f"파급 판정: {status_emojis[mc]}")
            elif mc == 1: st.warning(f"파급 판정: {status_emojis[mc]}")
            else: st.error(f"파급 판정: {status_emojis[mc]}")
            st.write(f"🟢 영향 미미: {pg:.1f}%")
            st.write(f"🟡 주의 필요: {py:.1f}%")
            st.write(f"🔴 심각한 영향: {pr:.1f}%")
            st.progress(int(pr))


# ══════════════════════════════════════════════
# [Frontend] 네트워크 맵 — 개선된 시각화
# ══════════════════════════════════════════════
st.divider()
st.subheader("🎯 이벤트 파급 경로 인과 네트워크 맵 (Interactive)")
st.markdown(
    "마우스 휠로 **확대/축소**, 노드 **드래그** 가능. "
    "NetCrafter $T_w$ 융합 가중치가 에지 두께에 실시간 반영됩니다."
)

net = Network(height="560px", width="100%", bgcolor="#1a1a2e",
              font_color="#ecf0f1", directed=True)

# 전체 엣지 가중치 수집 → 정규화 기준값 계산
all_display_weights = []
for s_n, p_n, o_n in raw_edges:
    if s_n not in node_to_idx or o_n not in node_to_idx: continue
    if s_n in excluded_nodes or o_n in excluded_nodes: continue
    si, oi = node_to_idx[s_n], node_to_idx[o_n]
    w = pure_adj[si, oi]
    if w > 0:
        all_display_weights.append(w)

# 정규화 기준: 95th percentile (최댓값 이상치 완화)
w_ref = np.percentile(all_display_weights, 95) if all_display_weights else 1.0

def scale_edge_width(raw_w, is_shock, mode, max_w):
    """엣지 두께 정규화 함수"""
    normalized = raw_w / w_ref  # 0~1 범주 정규화
    if mode == "로그 스케일 (분산 균형)":
        scaled = np.log1p(normalized * 9) / np.log1p(9)  # log 압축
    else:
        scaled = normalized
    width = scaled * max_w
    if is_shock:
        width = min(width * 1.8, max_w)  # 쇼크 강조 (상한 적용)
    return max(float(width), 0.5)

# 노드 추가
for node_name, idx in node_to_idx.items():
    clean = node_name
    for eng_k, kor_v in label_mapping.items():
        if node_name == eng_k:
            parts = kor_v.split(" ")
            clean = " ".join(parts[1:]) if len(parts) > 1 else kor_v

    if node_name in active_shocks:
        color, size = "#ff4b4b", 38
        label = f"🔥 {clean}\n[이벤트 발원지]"
        border = "#ff0000"
    elif node_name in excluded_nodes:
        color, size = "#4a4a6a", 14
        label = f"⚫ {clean}"
        border = "#666688"
    else:
        rr = probabilities[idx][2].item()
        yr = probabilities[idx][1].item()
        if rr > 0.45:
            color, size, border = "#c0392b", 30, "#e74c3c"
        elif rr > 0.25:
            color, size, border = "#e67e22", 25, "#f39c12"
        elif yr > 0.4:
            color, size, border = "#f1c40f", 22, "#f39c12"
        else:
            color, size, border = "#27ae60", 18, "#2ecc71"
        label = clean

    net.add_node(
        node_name, label=label, color={"background": color, "border": border},
        size=size, shape="dot",
        font={"size": 10, "color": "#ecf0f1"}
    )

# 엣지 추가 (정규화 두께 적용)
for s_n, p_n, o_n in raw_edges:
    if s_n not in node_to_idx or o_n not in node_to_idx: continue
    if s_n in excluded_nodes or o_n in excluded_nodes: continue
    si, oi = node_to_idx[s_n], node_to_idx[o_n]
    raw_w = pure_adj[si, oi]

    if raw_w < edge_threshold and not show_weak_edges:
        continue

    su = short_to_uri.get(s_n)
    ou = short_to_uri.get(o_n)
    tw_v = tw_map.get((su, ou), 0.0) if (su and ou) else 0.0
    display_w = raw_w * (1.0 + tw_v)

    is_shock_edge = (s_n in active_shocks) or \
                    (s_n == "Mat_Lithium" and "GPR_Channel" in active_shocks)
    e_width = scale_edge_width(display_w, is_shock_edge, edge_scale_mode, max_edge_width)

    if is_shock_edge:
        e_color = {"color": "#e74c3c", "opacity": 0.9}
    elif display_w > w_ref * 0.5:
        e_color = {"color": "#e67e22", "opacity": 0.7}  # 중간 강도 엣지
    else:
        e_color = {"color": "#5d6d7e", "opacity": 0.4}  # 약한 엣지 (투명도 처리)

    net.add_edge(
        s_n, o_n,
        value=e_width,
        color=e_color,
        title=(
            f"관계: {p_n}\n"
            f"BoK 가중치: {raw_w:.4f} | Tw 보정: {tw_v:.4f}\n"
            f"최종 에지 강도: {display_w:.4f}"
        ),
        arrows={"to": {"enabled": True, "scaleFactor": 0.5}}
    )

net.toggle_physics(True)
net.set_options("""
var options = {
  "physics": {
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -80,
      "centralGravity": 0.01,
      "springLength": 160,
      "springConstant": 0.06,
      "damping": 0.4,
      "avoidOverlap": 0.8
    },
    "minVelocity": 0.75,
    "stabilization": {"iterations": 200}
  },
  "edges": {
    "smooth": {"type": "dynamic"}
  }
}
""")

try:
    net.save_graph("pyvis_graph.html")
    with open("pyvis_graph.html", 'r', encoding='utf-8') as f:
        components.html(f.read(), height=580)
except Exception as e:
    st.error(f"네트워크 렌더링 오류: {e}")


# ══════════════════════════════════════════════
# [Frontend] Gemini 브리핑 + PDF 다운로드 (수정본)
# ══════════════════════════════════════════════
st.divider()
st.subheader("💡 이벤트 파급 영향 AI 브리핑")

# 세션 상태 변수 초기화 (중복 호출 및 화면 깜빡임 방지용)
if "briefing_text" not in st.session_state:
    st.session_state.briefing_text = None
if "last_shocks" not in st.session_state:
    st.session_state.last_shocks = []
if "briefing_loading" not in st.session_state:   # ← 추가
    st.session_state.briefing_loading = False     # ← 추가

# Tw 상위 엣지 추출 (Gemini·PDF 전달용)
tw_highlights = sorted(
    [
        (short_to_uri.get(s, s) or s,
         short_to_uri.get(o, o) or o,
         tw_map.get((short_to_uri.get(s), short_to_uri.get(o)), 0.0))
        for s, _, o in raw_edges
        if s not in excluded_nodes and o not in excluded_nodes
    ],
    key=lambda x: x[2], reverse=True
)[:10]

col_brief, col_pdf = st.columns([3, 1])

with col_brief:
    if gemini_api_key:
        # 조건이 바뀌었을 경우 이전 브리핑 결과 초기화
        if st.session_state.last_shocks != active_shocks:
            st.session_state.briefing_text = None
            st.session_state.last_shocks = active_shocks
        if st.button("🤖 AI 브리핑 생성/갱신하기", type="secondary",
                            disabled=st.session_state.briefing_loading):
            st.session_state.briefing_loading = True
            with st.spinner("🤖 Gemini가 실시간 검색 및 파급 영향을 분석 중입니다..."):
                st.session_state.briefing_text = generate_gemini_briefing(
                    api_key=gemini_api_key,
                    model=gemini_model,
                    active_shocks=active_shocks,
                    result_summary=result_summary,
                    bok_coef=bok_coef,
                    lithium_weight=lithium_weight,
                    tw_highlights=tw_highlights,
                )
            st.session_state.briefing_loading = False

        # 결과 출력
        if st.session_state.briefing_text:
            if st.session_state.briefing_text.startswith("⚠️"):
                st.error(st.session_state.briefing_text)
            else:
                st.markdown(st.session_state.briefing_text)
        else:
            st.caption("위 버튼을 누르면 실시간 검색 기반 AI 브리핑이 생성됩니다. (슬라이더 조작 시 자동 호출 차단)")

    else:
        # Gemini 미연동 시 정량 기반 기본 브리핑
        st.info("💡 Gemini API Key를 입력하면 AI 동적 브리핑이 활성화됩니다.")
        shock_str = ', '.join(active_shocks) if active_shocks else '없음'
        worst = max(result_summary.items(), key=lambda x: x[1]['pr'], default=(None, {}))
        st.markdown(f"""
- **분석 이벤트**: `{shock_str}`
- **산업연관 유효 가중치**: 리튬 가격 변동 → 배터리 수출단가 파급 강도 **{bok_coef * lithium_weight * 100:.4f}%**
- **가장 높은 파급 영향 품목**: {worst[0] or 'N/A'} — 심각 확률 {worst[1].get('pr', 0):.1f}%
- 사이드바 이벤트 체크박스 변경 시 네트워크 맵과 확률 카드가 즉시 재계산됩니다.
        """)

with col_pdf:
    st.markdown("#### 📄 보고서 출력")
    if st.button("PDF 보고서 생성", use_container_width=True, type="primary"):
        with st.spinner("PDF 생성 중..."):
            try:
                # 세션에 저장된 브리핑 텍스트를 PDF 전달용으로 사용
                pdf_bytes = generate_pdf_report(
                    active_shocks=active_shocks,
                    result_summary=result_summary,
                    briefing_text=st.session_state.briefing_text,
                    bok_coef=bok_coef,
                    lithium_weight=lithium_weight,
                    bom_coef=bom_coef,
                    tw_highlights=tw_highlights,
                )
                st.download_button(
                    label="⬇️ PDF 다운로드",
                    data=pdf_bytes,
                    file_name=f"KEWS_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ 보고서 생성 완료!")
            except Exception as e:
                st.error(f"PDF 생성 오류: {e}")