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
import plotly.graph_objects as go
from collections import deque
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
    result_summary: dict,
    bok_coef: float,
    lithium_weight: float,
    tw_highlights: list,
) -> str:
    if not api_key:
        return None

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
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096}
    }
    payload["tools"] = [{"googleSearch": {}}]
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        candidate = resp.json()["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
        finish_reason = candidate.get("finishReason", "")
        if finish_reason == "MAX_TOKENS":
            text += "\n\n⚠️ (응답이 토큰 제한으로 중간에 잘렸습니다. maxOutputTokens를 늘려주세요.)"
        return text
    except Exception as e:
        return f"⚠️ Gemini API 오류: {e}"


# ══════════════════════════════════════════════
# [Backend 0-1] Gemini Grounding 기반 실시간 통계 수집
# ══════════════════════════════════════════════
def fetch_real_trade_weight(api_key: str, model: str) -> float:
    if not api_key:
        return 0.05

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
                wait = 2 ** attempt
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
# [Backend 1] NetCrafter Tw 계산 엔진
# ══════════════════════════════════════════════
def calculate_ontology_tw(g, KEWS, trade_weight=0.05):
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
def load_kg_with_tw_fusion(ttl_path, bok_coef, lithium_weight, bom_coef, trade_weight=0.05):
    g = Graph()
    g.parse(ttl_path, format="turtle")
    kews = Namespace("http://k-ews.org/ontology#")

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
# [NEW] K-SCI (한국전략광물지수) 계산 엔진
# ══════════════════════════════════════════════
# 컬럼명은 실제 CSV 스키마에 맞춰 수정 필요
KSCI_COLUMNS = {
    "gpr": "GPR",
    "gscpi": "GSCPI",
    "hhi_li": "HHI_탄산리튬",      # 품목별 HHI — 가중평균으로 합성
    "hhi_gr": "HHI_흑연_기타",
    "price_li": "UP_탄산리튬",
    "price_gr": "UP_흑연_기타",
    "vol_chg_li": "VOL_CHG_탄산리튬",
    "vol_chg_gr": "VOL_CHG_흑연_기타",
    "ppi": "PPI_리튬이온",
}

def calculate_ksci(csv_path, target_date,
                    w_li=0.85, w_gr=0.15, alpha=0.2,
                    cols=KSCI_COLUMNS):
    try:
        df = pd.read_csv(csv_path)
        df['YearMonth'] = pd.to_datetime(df['YearMonth'])

        def zscore(series):
            s = pd.to_numeric(series, errors='coerce')
            std = s.std()
            return (s - s.mean()) / std if std and std > 0 else s * 0.0

        gpr_col   = cols.get("gpr")
        gscpi_col = cols.get("gscpi")

        z_gpr   = zscore(df[gpr_col])   if gpr_col   in df.columns else pd.Series(0.0, index=df.index)
        z_gscpi = zscore(df[gscpi_col]) if gscpi_col in df.columns else pd.Series(0.0, index=df.index)

        # ── HHI 보정: 탄산리튬/기타흑연 HHI를 K-SCI 가중치(0.85/0.15)로 합성 ──
        hhi_li_col = cols.get("hhi_li")
        hhi_gr_col = cols.get("hhi_gr")
        if hhi_li_col in df.columns and hhi_gr_col in df.columns:
            hhi_blend = w_li * df[hhi_li_col] + w_gr * df[hhi_gr_col]
            z_hhi = zscore(hhi_blend)
        else:
            z_hhi = pd.Series(0.0, index=df.index)

        raw = (w_li * z_gpr + w_gr * z_gscpi) * (1 + alpha * z_hhi)
        mn, mx = raw.min(), raw.max()
        ksci_series = ((raw - mn) / (mx - mn) * 100) if mx > mn else raw * 0.0
        df['K_SCI'] = ksci_series

        target_dt = pd.to_datetime(target_date)
        row = df[(df['YearMonth'].dt.year == target_dt.year) &
                  (df['YearMonth'].dt.month == target_dt.month)]
        if row.empty:
            row = df.iloc[[-1]]

        ksci_value = float(row['K_SCI'].iloc[0])
        return ksci_value, df[['YearMonth', 'K_SCI']]

    except Exception:
        return 0.0, None


def get_ksci_stage(ksci_value):
    """K-SCI 점수를 3단계 신호등으로 변환"""
    if ksci_value < 20:
        return "🟢 안정", "관심 (Blue)", "#27ae60"
    elif ksci_value < 40:
        return "🟡 주의", "주의 (Yellow)", "#f39c12"
    else:
        return "🔴 위험", "경계/심각", "#e74c3c"


# ══════════════════════════════════════════════
# [NEW] Layer 2 복합 검증 (오경보 방지 이중 필터)
# ══════════════════════════════════════════════
def calculate_layer2_status(csv_path, target_date, cols=KSCI_COLUMNS):
    """
    가격 플래그: 탄산리튬/기타흑연 Z>1 또는 전월비 +10% 이상
    물동량 플래그: 탄산리튬/기타흑연 물동량 3개월 이동평균 -10% 이하
    PPI 플래그: PPI_리튬이온 전월비 +1% 이상
    """
    result = {"price_flag": False, "volume_flag": False, "ppi_flag": False, "final": False}
    try:
        df = pd.read_csv(csv_path)
        df['YearMonth'] = pd.to_datetime(df['YearMonth'])
        target_dt = pd.to_datetime(target_date)
        match = df[(df['YearMonth'].dt.year == target_dt.year) &
                    (df['YearMonth'].dt.month == target_dt.month)]
        if match.empty:
            return result
        idx = match.index[0]

        # 가격 플래그
        for key in ("price_li", "price_gr"):
            c = cols.get(key)
            if c and c in df.columns:
                series = pd.to_numeric(df[c], errors='coerce')
                m, s = series.mean(), series.std()
                z = (series.iloc[idx] - m) / s if s and s > 0 else 0
                pct_chg = series.pct_change().iloc[idx]
                if (z is not None and z > 1) or (pct_chg is not None and pct_chg >= 0.10):
                    result["price_flag"] = True

        # 물동량 플래그
        for key in ("vol_chg_li", "vol_chg_gr"):
            c = cols.get(key)
            if c and c in df.columns:
                series = pd.to_numeric(df[c], errors='coerce')
                ma3 = series.rolling(3).mean()
                if idx < len(ma3) and pd.notna(ma3.iloc[idx]) and ma3.iloc[idx] <= -0.10:
                    result["volume_flag"] = True

        # PPI 플래그
        ppi_c = cols.get("ppi")
        if ppi_c and ppi_c in df.columns:
            series = pd.to_numeric(df[ppi_c], errors='coerce')
            ppi_pct = series.pct_change().iloc[idx]
            if ppi_pct is not None and ppi_pct >= 0.01:
                result["ppi_flag"] = True

        # 최종 확정: 3개 플래그 중 하나 이상 발동 시 실물 충격 동반으로 판단
        result["final"] = result["price_flag"] or result["volume_flag"] or result["ppi_flag"]
        return result

    except Exception:
        return result


# ══════════════════════════════════════════════
# [NEW] GNN 확률 → 소부장 4단계 위기경보 매핑
# ══════════════════════════════════════════════
def map_to_sobujang_stage(pg, py, pr, layer2_final):
    """
    pg, py, pr: 0~1 스케일의 Green/Yellow/Red 확률
    반환: (K-EWS 출력 상태 라벨, 소부장 법정 단계, 색상코드)
    """
    if pg >= 0.60:
        return "🟢 안정 (Green)", "관심 (Blue)", "#27ae60"
    if pr >= 0.30:
        if layer2_final:
            return "🔴 위험 (Red) - 확정", "심각 (Red)", "#c0392b"
        else:
            return "🔴 위험 (Red) - 예비", "경계 (Orange)", "#e67e22"
    if py >= 0.35 or pr < 0.30:
        return "🟡 주의 (Yellow)", "주의 (Yellow)", "#f1c40f"
    return "🟡 주의 (Yellow)", "주의 (Yellow)", "#f1c40f"


# ══════════════════════════════════════════════
# [NEW] 백테스팅 / OOS 검증 결과 (보고서 2-b-iv 고정 결과)
# ══════════════════════════════════════════════
BACKTEST_EVENTS = [
    {"event": "미중무역전쟁",      "date": "2018-07", "ksci": 19.3,  "actual": "안정", "predicted": "안정", "correct": True},
    {"event": "일본수출규제",      "date": "2019-07", "ksci": 14.7,  "actual": "안정", "predicted": "안정", "correct": True},
    {"event": "코로나충격",        "date": "2020-03", "ksci": 7.9,   "actual": "안정", "predicted": "안정", "correct": True},
    {"event": "리튬가격폭등",      "date": "2022-01", "ksci": 35.9,  "actual": "위험", "predicted": "주의", "correct": False},
    {"event": "러-우전쟁",         "date": "2022-03", "ksci": 100.0, "actual": "위험", "predicted": "위험", "correct": True},
    {"event": "흑연수출통제발표",  "date": "2023-10", "ksci": 52.2,  "actual": "위험", "predicted": "위험", "correct": True},
    {"event": "흑연통제실효",      "date": "2024-01", "ksci": 33.9,  "actual": "주의", "predicted": "주의", "correct": True},
]
BACKTEST_METRICS = {
    "accuracy": 0.857, "f1": 0.571,
    "oos_recall": 1.000, "oos_precision": 0.333,
    "train_period": "2015-01 ~ 2022-12",
    "test_period": "2023-01 ~ 2025-12",
}


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

    font_name = "Helvetica"
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

    story.append(Paragraph("K-EWS 공급망 이벤트 파급 영향 분석 보고서", s_title))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#2980b9")))
    story.append(Spacer(1, 4*mm))

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
    for i, c in enumerate(row_colors):
        tbl_style.add("BACKGROUND", (1, i+1), (1, i+1), c)
        tbl_style.add("TEXTCOLOR",  (1, i+1), (1, i+1), colors.white)
    tbl.setStyle(tbl_style)
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

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

    story.append(Paragraph(f"③ AI 파급 영향 브리핑 (Gemini)", s_h2))
    brief = briefing_text or "Gemini API가 연동되지 않은 상태입니다."
    for line in brief.split("\n"):
        line = line.strip()
        if line:
            story.append(Paragraph(line, s_briefing))
    story.append(Spacer(1, 6*mm))

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
# [NEW] BFS 경로 필터 — 발원지→타깃 사이 노드만 추출
# ══════════════════════════════════════════════
def get_path_nodes_and_edges(raw_edges, sources, targets, excluded):
    """
    BFS로 sources → targets 사이에 존재하는 노드·엣지만 반환.
    sources: 쇼크 발원 노드 집합
    targets: 결과 카드 노드 집합 (label_mapping 키)
    """
    # 정방향 인접 dict
    adj_fwd = {}
    for s, p, o in raw_edges:
        if s in excluded or o in excluded:
            continue
        adj_fwd.setdefault(s, []).append((o, p))

    # BFS: sources 에서 출발하여 도달 가능한 노드 전부
    reachable_from_src = set(sources)
    q = deque(sources)
    visited = set(sources)
    while q:
        cur = q.popleft()
        for nxt, _ in adj_fwd.get(cur, []):
            if nxt not in visited:
                visited.add(nxt)
                reachable_from_src.add(nxt)
                q.append(nxt)

    # 역방향 인접 dict
    adj_bwd = {}
    for s, p, o in raw_edges:
        if s in excluded or o in excluded:
            continue
        adj_bwd.setdefault(o, []).append((s, p))

    # BFS: targets 에서 역방향으로 도달 가능한 노드 전부
    reachable_to_tgt = set(targets)
    q = deque(targets)
    visited2 = set(targets)
    while q:
        cur = q.popleft()
        for prv, _ in adj_bwd.get(cur, []):
            if prv not in visited2:
                visited2.add(prv)
                reachable_to_tgt.add(prv)
                q.append(prv)

    # 교집합 = 경로상 노드
    path_nodes = reachable_from_src & reachable_to_tgt

    # 경로상 엣지만 필터
    path_edges = [
        (s, p, o) for s, p, o in raw_edges
        if s in path_nodes and o in path_nodes
        and s not in excluded and o not in excluded
    ]

    return path_nodes, path_edges


# ══════════════════════════════════════════════
# [NEW] 4단계 레이어 정의 (온톨로지 인과 구조 기반)
# ══════════════════════════════════════════════
#
# 실제 KG 인과 방향 (StatisticalRelation hasSource→hasTarget):
#
#  Layer 0  매크로 쇼크 채널
#    GPR_Channel, GSCPI_Channel, Event_COVID19
#
#  Layer 1  원자재 (쇼크 직접 수신)
#    Mat_Lithium, Mat_Graphite_Other
#    (GPR/GSCPI → LONG_TERM_CAUSES/LEADS_PRICE → 여기)
#
#  Layer 2  중간재/공정
#    Mat_NCM_Cathode   (Mat_Lithium → PHYSICAL_INPUT_TO → 여기)
#    Mat_Anode_Active  (Mat_Graphite_Other → PHYSICAL_INPUT_TO → 여기)
#
#  Layer 3  최종 지표
#    PPI_리튬이온       (중간재·원자재 → LONG_TERM_CAUSES/EXPLAINS_VARIANCE → 여기)
#    Bat_Export_UnitPrice (중간재·원자재·PPI → LONG_TERM_CAUSES/VALUE_CHAIN_FORWARD → 여기)

LAYER_MAP = {
    # Layer 0 – 매크로 쇼크
    "GPR_Channel":          0,
    "GSCPI_Channel":        0,
    "Event_COVID19":        0,
    # Layer 1 – 원자재
    "Mat_Lithium":          1,
    "Mat_Graphite_Other":   1,
    # Layer 2 – 중간재
    "Mat_NCM_Cathode":      2,
    "Mat_Anode_Active":     2,
    # Layer 3 – 최종 지표
    "PPI_리튬이온":          3,
    "Bat_Export_UnitPrice": 3,
}

# 레이어 라벨 (헤더 표시용)
LAYER_LABELS = {
    0: "매크로 쇼크",
    1: "원자재",
    2: "중간재",
    3: "최종 지표",
}

KOR_LABEL = {
    "GPR_Channel":          "지정학리스크\n(GPR)",
    "GSCPI_Channel":        "글로벌공급망\n압력(GSCPI)",
    "Mat_Graphite_Other":   "흑연\n(Mat_Graphite)",
    "Event_COVID19":        "팬데믹\n쇼크",
    "Mat_Lithium":          "탄산리튬\n(Mat_Lithium)",
    "Mat_NCM_Cathode":      "NCM 양극재\n(중간재)",
    "Mat_Anode_Active":     "인조흑연\n음극활물질(중간재)",
    "PPI_리튬이온":          "배터리 PPI\n(최종)",
    "Bat_Export_UnitPrice": "배터리\n수출단가(최종)",
}


# ══════════════════════════════════════════════
# [NEW] Sankey 다이어그램 생성 (4단계 레이어 기반)
# ══════════════════════════════════════════════
def build_sankey(
    path_nodes, path_edges,
    active_shocks, label_mapping, excluded_nodes,
    pure_adj, node_to_idx, tw_map, short_to_uri,
    probabilities
):
    """
    온톨로지 인과 구조 기반 4단계 Sankey:
      Layer0(매크로쇼크) → Layer1(원자재) → Layer2(중간재) → Layer3(최종지표)
    """
    if not path_nodes or not path_edges:
        return None

    shock_nodes = set(active_shocks)

    # 경로상 노드 중 LAYER_MAP에 있는 것만 사용
    # (LAYER_MAP에 없는 보조 노드는 Sankey에서 제외해 깔끔하게)
    sankey_nodes = [n for n in path_nodes if n in LAYER_MAP]
    # 발원지가 LAYER_MAP에 없을 수도 있으니 보장
    for s in shock_nodes:
        if s not in sankey_nodes and s in path_nodes:
            sankey_nodes.append(s)

    if not sankey_nodes:
        return None

    node_idx_s = {n: i for i, n in enumerate(sankey_nodes)}

    # ── 노드 색상 ──
    def node_color(n):
        if n in shock_nodes:
            return "rgba(231,76,60,0.90)"
        idx = node_to_idx.get(n)
        if idx is None:
            return "rgba(100,100,100,0.6)"
        pr = probabilities[idx][2].item()
        py = probabilities[idx][1].item()
        if pr > 0.45:
            return "rgba(192,57,43,0.90)"
        elif pr > 0.25:
            return "rgba(230,126,34,0.85)"
        elif py > 0.4:
            return "rgba(241,196,15,0.85)"
        else:
            return "rgba(39,174,96,0.85)"

    # ── 노드 hover 텍스트 ──
    def node_hover(n):
        idx = node_to_idx.get(n)
        layer = LAYER_MAP.get(n, "?")
        layer_str = LAYER_LABELS.get(layer, str(layer))
        base = f"<b>{KOR_LABEL.get(n, n)}</b><br>레이어: {layer_str}"
        if idx is not None:
            pg = probabilities[idx][0].item() * 100
            py = probabilities[idx][1].item() * 100
            pr = probabilities[idx][2].item() * 100
            base += (f"<br>🟢 영향 미미: {pg:.1f}%"
                     f"<br>🟡 주의 필요: {py:.1f}%"
                     f"<br>🔴 심각한 영향: {pr:.1f}%")
        return base

    node_colors  = [node_color(n) for n in sankey_nodes]
    node_labels  = [KOR_LABEL.get(n, n) for n in sankey_nodes]
    node_hovers  = [node_hover(n) for n in sankey_nodes]

    # ── x/y 좌표: 레이어별 균등 배치 ──
    layer_members = {0: [], 1: [], 2: [], 3: []}
    for n in sankey_nodes:
        l = LAYER_MAP.get(n, 1)
        layer_members[l].append(n)

    x_layer = {0: 0.01, 1: 0.34, 2: 0.67, 3: 0.99}
    node_xy = {}
    for l, members in layer_members.items():
        x = x_layer[l]
        for rank, n in enumerate(members):
            y = (rank + 1) / (len(members) + 1)
            node_xy[n] = (x, y)

    xs = [node_xy.get(n, (0.5, 0.5))[0] for n in sankey_nodes]
    ys = [node_xy.get(n, (0.5, 0.5))[1] for n in sankey_nodes]

    # ── 엣지 (방향: layer 증가 방향만 유효) ──
    src_list, tgt_list, val_list, ecol_list, elabel_list = [], [], [], [], []

    # 최대 가중치 기준 계산
    valid_ws = []
    for s, p, o in path_edges:
        if s not in node_idx_s or o not in node_idx_s:
            continue
        si = node_to_idx.get(s)
        oi = node_to_idx.get(o)
        if si is None or oi is None:
            continue
        valid_ws.append(float(pure_adj[si, oi]))
    w_ref_s = max(valid_ws) if valid_ws else 1.0
    if w_ref_s == 0:
        w_ref_s = 1.0

    for s, p, o in path_edges:
        if s not in node_idx_s or o not in node_idx_s:
            continue
        # 레이어 방향 검사: source 레이어 ≤ target 레이어여야 순방향
        sl = LAYER_MAP.get(s, -1)
        ol = LAYER_MAP.get(o, -1)
        if sl > ol:   # 역방향(피드백) 엣지는 Sankey에서 제외
            continue

        si = node_to_idx.get(s)
        oi = node_to_idx.get(o)
        if si is None or oi is None:
            continue

        raw_w = float(pure_adj[si, oi])
        su = short_to_uri.get(s)
        ou = short_to_uri.get(o)
        tw_v = tw_map.get((su, ou), 0.0) if (su and ou) else 0.0
        display_w = raw_w * (1.0 + tw_v)
        norm_w = max(display_w / w_ref_s * 40, 1.0)

        is_shock = s in shock_nodes
        if is_shock:
            e_color = "rgba(231,76,60,0.50)"
        elif sl == 1:   # 원자재 → 중간재
            e_color = "rgba(230,126,34,0.45)"
        else:           # 중간재 → 최종
            e_color = "rgba(52,152,219,0.40)"

        src_list.append(node_idx_s[s])
        tgt_list.append(node_idx_s[o])
        val_list.append(norm_w)
        ecol_list.append(e_color)
        elabel_list.append(
            f"<b>{KOR_LABEL.get(s,s)} → {KOR_LABEL.get(o,o)}</b><br>"
            f"관계: {p}<br>"
            f"BoK 가중치: {raw_w:.4f}<br>"
            f"Tw 보정: {tw_v:.4f}<br>"
            f"최종 강도: {display_w:.4f}"
        )

    if not src_list:
        return None

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=25,
            thickness=22,
            line=dict(color="rgba(255,255,255,0.3)", width=0.8),
            label=node_labels,
            color=node_colors,
            x=xs,
            y=ys,
            customdata=node_hovers,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=src_list,
            target=tgt_list,
            value=val_list,
            color=ecol_list,
            label=elabel_list,
            hovertemplate="%{label}<extra></extra>",
        )
    ))

    # 레이어 헤더 annotation
    annotations = []
    for l, label_txt in LAYER_LABELS.items():
        annotations.append(dict(
            x=x_layer[l], y=1.06,
            xref="paper", yref="paper",
            text=f"<b>【{label_txt}】</b>",
            showarrow=False,
            font=dict(size=12, color="#ecf0f1"),
            xanchor="center",
        ))

    fig.update_layout(
        title_text=(
            "파급 경로 Sankey  │  "
            "🔴 매크로쇼크 → 🟠 원자재 → 🔵 중간재 → 최종지표"
        ),
        title_font=dict(size=13, color="#ecf0f1"),
        font_size=11,
        paper_bgcolor="#1a1a2e",
        font_color="#ecf0f1",
        height=500,
        margin=dict(l=20, r=20, t=70, b=20),
        annotations=annotations,
    )
    return fig


# ══════════════════════════════════════════════
# [NEW] 계층형 네트워크 맵 (vis.js hierarchical, 4단계 레이어)
# ══════════════════════════════════════════════
def build_hierarchical_network(
    path_nodes, path_edges,
    active_shocks, label_mapping, excluded_nodes,
    pure_adj, node_to_idx, tw_map, short_to_uri,
    probabilities, top_k, edge_threshold, max_edge_width,
    show_weak_edges
):
    net = Network(height="560px", width="100%", bgcolor="#1a1a2e",
                  font_color="#ecf0f1", directed=True)

    shock_nodes = set(active_shocks)

    # ── 노드 추가 (4단계 level) ──
    for node_name in path_nodes:
        clean = KOR_LABEL.get(node_name, node_name).replace("\n", " ")
        idx   = node_to_idx.get(node_name)

        # level: LAYER_MAP 우선, 없으면 추정
        level = LAYER_MAP.get(node_name, 1) + 1   # vis.js level은 1부터

        if node_name in shock_nodes:
            color  = "#ff4b4b"
            border = "#ff0000"
            size   = 36
            label  = f"🔥 {clean}"
            title  = (f"<b>{clean}</b><br>"
                      f"<i>이벤트 발원지 (Layer 0)</i>")
        elif idx is not None:
            rr = probabilities[idx][2].item()
            yr = probabilities[idx][1].item()
            pg = probabilities[idx][0].item() * 100
            py = yr * 100
            pr = rr * 100
            if rr > 0.45:
                color, size, border = "#c0392b", 30, "#e74c3c"
            elif rr > 0.25:
                color, size, border = "#e67e22", 26, "#f39c12"
            elif yr > 0.4:
                color, size, border = "#d4ac0d", 22, "#f1c40f"
            else:
                color, size, border = "#27ae60", 18, "#2ecc71"
            label = clean
            layer_name = LAYER_LABELS.get(LAYER_MAP.get(node_name, 1), "")
            title = (
                f"<b>{clean}</b>"
                f"<br><i>레이어: {layer_name}</i>"
                f"<br>🟢 영향 미미: {pg:.1f}%"
                f"<br>🟡 주의 필요: {py:.1f}%"
                f"<br>🔴 심각한 영향: {pr:.1f}%"
            )
        else:
            color, size, border = "#5d6d7e", 16, "#7f8c8d"
            label = clean
            title = f"<b>{clean}</b>"

        net.add_node(
            node_name,
            label=label,
            color={"background": color, "border": border},
            size=size, shape="dot", level=level,
            font={"size": 11, "color": "#ecf0f1"},
            title=title,
        )

    # ── Top-K 필터 & 엣지 추가 ──
    edge_candidates = []
    for s, p, o in path_edges:
        if s not in path_nodes or o not in path_nodes:
            continue
        si = node_to_idx.get(s)
        oi = node_to_idx.get(o)
        if si is None or oi is None:
            continue
        raw_w = float(pure_adj[si, oi])
        su = short_to_uri.get(s)
        ou = short_to_uri.get(o)
        tw_v = tw_map.get((su, ou), 0.0) if (su and ou) else 0.0
        display_w = raw_w * (1.0 + tw_v)
        edge_candidates.append((s, p, o, raw_w, tw_v, display_w))

    edge_candidates.sort(key=lambda x: x[5], reverse=True)
    if top_k > 0:
        edge_candidates = edge_candidates[:top_k]

    w_ref = max((e[5] for e in edge_candidates), default=1.0)
    if w_ref == 0:
        w_ref = 1.0

    for s, p, o, raw_w, tw_v, display_w in edge_candidates:
        if raw_w < edge_threshold and not show_weak_edges:
            continue

        norm   = display_w / w_ref
        scaled = np.log1p(norm * 9) / np.log1p(9)
        e_width = max(float(scaled * max_edge_width), 0.5)

        # 레이어 간 색상 구분
        sl = LAYER_MAP.get(s, 1)
        ol = LAYER_MAP.get(o, 1)
        if s in shock_nodes:
            e_color = {"color": "#e74c3c", "opacity": 0.95}   # 쇼크→원자재: 빨강
        elif sl == 1 and ol == 2:
            e_color = {"color": "#e67e22", "opacity": 0.85}   # 원자재→중간재: 주황
        elif sl <= 2 and ol == 3:
            e_color = {"color": "#3498db", "opacity": 0.80}   # 중간재→최종: 파랑
        else:
            e_color = {"color": "#5d6d7e", "opacity": 0.50}   # 기타: 회색

        net.add_edge(
            s, o,
            value=e_width,
            color=e_color,
            title=(
                f"<b>관계: {p}</b><br>"
                f"BoK: {raw_w:.4f} | Tw: {tw_v:.4f}<br>"
                f"최종 강도: {display_w:.4f}"
            ),
            arrows={"to": {"enabled": True, "scaleFactor": 0.7}},
        )

    # ── vis.js 계층형 레이아웃 옵션 ──
    net.set_options("""
var options = {
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "LR",
      "sortMethod": "directed",
      "levelSeparation": 240,
      "nodeSpacing": 110,
      "treeSpacing": 180,
      "blockShifting": true,
      "edgeMinimization": true
    }
  },
  "physics": { "enabled": false },
  "edges": {
    "smooth": { "type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.4 }
  },
  "interaction": {
    "tooltipDelay": 80,
    "hover": true,
    "navigationButtons": true
  }
}
""")
    return net


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

#### 🧬 2. NetCrafter 기반 Tw 융합 원리
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
show_weak_edges = st.sidebar.checkbox("약한 엣지 표시 (임계값 이하)", value=False)
edge_threshold  = st.sidebar.slider("엣지 표시 임계값 (가중치)", 0.0, 1.0, 0.05, step=0.01)
max_edge_width  = st.sidebar.slider("최대 엣지 두께 상한", 3.0, 20.0, 8.0, step=1.0)

# ── [NEW] 경로 필터 옵션 ──
st.sidebar.header("🔍 5. 경로 필터 설정")
path_only_mode = st.sidebar.checkbox(
    "발원지→결과 경로 노드만 표시",
    value=True,
    help="체크 시 쇼크 발원지에서 결과 카드 노드까지의 경로상 노드·엣지만 표시합니다."
)
top_k_edges = st.sidebar.slider(
    "Top-K 엣지만 표시 (Tw 기준, 0=전체)",
    min_value=0, max_value=50, value=20, step=5,
    help="Tw 가중치 기준 상위 K개 엣지만 그립니다. 0으로 설정하면 전체 표시."
)

# ══════════════════════════════════════════════
# [Engine] 백엔드 실행
# ══════════════════════════════════════════════
TTL_PATH = "K_EWS_KnowledgeGraph7.ttl"
CSV_PATH = "kews_preprocessed_132m.csv"

if "dynamic_trade_weight" not in st.session_state:
    st.session_state.dynamic_trade_weight = 0.05
if "trade_weight_fetched" not in st.session_state:
    st.session_state.trade_weight_fetched = False

if gemini_api_key and not st.session_state.trade_weight_fetched:
    with st.spinner("🔍 Gemini가 관세청/구글 통계에서 리튬 실제 무역 비중 가중치를 실시간 채굴 중..."):
        mined_w = fetch_real_trade_weight(gemini_api_key, gemini_model)
        st.session_state.dynamic_trade_weight = mined_w
        st.session_state.trade_weight_fetched = True

st.sidebar.info(f"🧬 Gemini 동적 마이닝 가중치: {st.session_state.dynamic_trade_weight:.5f}")

(weighted_adj, node_list, node_to_idx,
 excluded_nodes, raw_edges, pure_adj,
 tw_map, short_to_uri) = load_kg_with_tw_fusion(
    TTL_PATH, bok_coef, lithium_weight, bom_coef,
    trade_weight=st.session_state.dynamic_trade_weight
)

num_nodes      = len(node_list)
dynamic_labels = generate_labels(CSV_PATH, "2023-10-01", node_to_idx)

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

    for i in range(num_nodes):
        incoming = torch.dot(weighted_adj[i], shock_mask.squeeze())
        if incoming > 0:
            logits[i][0] -= incoming * alpha_green
            logits[i][1] += incoming * (alpha_red * 0.6)
            logits[i][2] += incoming * alpha_red

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
    "Mat_RareEarth":       "💎 희토류",
}
status_emojis = {0: "🟢 영향 미미", 1: "🟡 주의 필요", 2: "🔴 심각한 영향"}

result_summary = {}
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
# [NEW] K-SCI 종합 지수 + 소부장 4단계 매핑 패널
# ══════════════════════════════════════════════
st.divider()
st.subheader("🚦 K-SCI 종합 위기 지수 & 소부장 법정 경보 매핑")

TARGET_DATE_FOR_KSCI = "2023-10-01"  # generate_labels와 동일 기준 시점, 필요시 사이드바로 노출 가능

ksci_value, ksci_history = calculate_ksci(CSV_PATH, TARGET_DATE_FOR_KSCI)
ksci_stage_label, ksci_gov_label, ksci_color = get_ksci_stage(ksci_value)
layer2_result = calculate_layer2_status(CSV_PATH, TARGET_DATE_FOR_KSCI)

col_ksci, col_l2 = st.columns([1, 1])

with col_ksci:
    st.markdown(f"#### {ksci_stage_label}  ·  K-SCI {ksci_value:.1f}")
    st.markdown(f"**소부장 법정 매칭 단계**: {ksci_gov_label}")
    st.progress(min(int(ksci_value), 100))
    st.caption(
        "0\~20: 안정(관심) │ 20\~40: 주의 │ 40\~100: 위험  "
        "(분포 백테스팅 기준 임계값)"
    )
    if ksci_history is None:
        st.warning("⚠️ CSV에서 GPR/GSCPI 컬럼을 찾지 못해 K-SCI가 중립값(0)으로 표시됩니다. "
                    "`KSCI_COLUMNS` 딕셔너리의 컬럼명을 실제 데이터에 맞게 수정하세요.")

with col_l2:
    st.markdown("#### Layer 2 실물 검증 플래그")
    st.write(f"{'✅' if layer2_result['price_flag'] else '⬜'} 가격 플래그 (Z>1 또는 전월비 +10%↑)")
    st.write(f"{'✅' if layer2_result['volume_flag'] else '⬜'} 물동량 플래그 (3MA -10%↓)")
    st.write(f"{'✅' if layer2_result['ppi_flag'] else '⬜'} PPI 플래그 (전월비 +1%↑)")
    if layer2_result["final"]:
        st.error("L2 플래그 발동 → K-SCI 위험 판정 시 '심각' 확정 가능")
    else:
        st.info("L2 플래그 미발동 → K-SCI 위험 판정도 '경계(예비)'로 강등")

st.markdown("##### 품목별 GNN 확률 → 소부장 4단계 매핑")
sobu_cols = st.columns(len(label_mapping))
for col, (eng_id, kor_name) in zip(sobu_cols, label_mapping.items()):
    with col:
        if eng_id not in node_to_idx or eng_id in excluded_nodes:
            st.caption(f"{kor_name}: 매핑 제외")
            continue
        idx = node_to_idx[eng_id]
        pg_ = probabilities[idx][0].item()
        py_ = probabilities[idx][1].item()
        pr_ = probabilities[idx][2].item()
        label, gov_stage, color = map_to_sobujang_stage(pg_, py_, pr_, layer2_result["final"])
        st.markdown(f"**{kor_name}**")
        st.markdown(
            f"<span style='color:{color}; font-weight:bold;'>{label}</span><br>"
            f"<span style='font-size:0.85em;'>법정 단계: {gov_stage}</span>",
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════
# [NEW] 백테스팅 / OOS 검증 결과
# ══════════════════════════════════════════════
with st.expander("📋 모델 신뢰도 — 백테스팅 & OOS 검증 결과 (2015~2025)", expanded=False):
    bt_df = pd.DataFrame(BACKTEST_EVENTS)
    bt_df_display = bt_df.rename(columns={
        "event": "사건", "date": "시점", "ksci": "K-SCI 점수",
        "actual": "실제 레이블", "predicted": "모델 예측", "correct": "정탐 여부"
    })
    bt_df_display["정탐 여부"] = bt_df_display["정탐 여부"].map({True: "✅", False: "❌"})
    st.dataframe(bt_df_display, use_container_width=True, hide_index=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Full-sample 정확도", f"{BACKTEST_METRICS['accuracy']*100:.1f}%")
    m2.metric("Full-sample F1", f"{BACKTEST_METRICS['f1']:.3f}")
    m3.metric("OOS Recall", f"{BACKTEST_METRICS['oos_recall']:.3f}")
    m4.metric("OOS Precision", f"{BACKTEST_METRICS['oos_precision']:.3f}")
    st.caption(
        f"Train: {BACKTEST_METRICS['train_period']}  |  "
        f"Test: {BACKTEST_METRICS['test_period']}  |  "
        "OOS Precision이 낮은 이유: 2025-03(미국-이란 핵협상 교착), "
        "2025-06(이란 핵시설 공습) 시점에 GPR만 단독 반응한 오경보 2건 포함"
    )

# ══════════════════════════════════════════════
# [Frontend] 파급 경로 시각화 (탭 전환)
# ══════════════════════════════════════════════
st.divider()
st.subheader("🎯 이벤트 파급 경로 시각화")

# BFS 경로 필터 공통 계산
target_node_ids = set(label_mapping.keys())
shock_node_ids  = set(active_shocks)

if path_only_mode and shock_node_ids:
    path_nodes, path_edges = get_path_nodes_and_edges(
        raw_edges, shock_node_ids, target_node_ids, excluded_nodes
    )
    # 발원지·타깃 노드는 반드시 포함
    path_nodes = path_nodes | shock_node_ids | target_node_ids
else:
    path_nodes = set(n for n in node_to_idx if n not in excluded_nodes)
    path_edges = [(s, p, o) for s, p, o in raw_edges
                  if s not in excluded_nodes and o not in excluded_nodes]

# ─── 탭 구성 ───
tab_sankey, tab_hier, tab_full = st.tabs([
    "📊 Sankey 흐름도  (직관적 파급 경로)",
    "🗂️ 계층형 네트워크  (레이어 분리 맵)",
    "🌐 전체 네트워크  (원본 pyvis)",
])

# ── TAB 1: Sankey ──
with tab_sankey:
    st.markdown(
        "**매크로쇼크(GPR/GSCPI) → 원자재(리튬·흑연) → 중간재(양극재·음극재) → 최종지표(PPI·수출단가)**"
        " 4단계 인과 흐름을 좌→우로 표시합니다.  \n"
        "링크 두께 = Tw 보정 파급 강도 | 노드 색 = GNN 위험도 판정 "
        "| 노드·링크에 마우스를 올리면 상세 수치가 표시됩니다."
    )
    sankey_fig = build_sankey(
        path_nodes, path_edges,
        active_shocks, label_mapping, excluded_nodes,
        pure_adj, node_to_idx, tw_map, short_to_uri,
        probabilities
    )
    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)
    else:
        st.info("현재 선택된 쇼크 시나리오에서 경로 데이터가 없습니다. 사이드바에서 이벤트를 체크해 주세요.")

# ── TAB 2: 계층형 네트워크 ──
with tab_hier:
    st.markdown(
        "**L0(매크로쇼크)** → **L1(원자재)** → **L2(중간재)** → **L3(최종지표)** "
        "4단계 레이어를 좌→우로 배치합니다.  \n"
        "엣지 색: 🔴 쇼크→원자재 &nbsp;|&nbsp; 🟠 원자재→중간재 &nbsp;|&nbsp; 🔵 중간재→최종지표  \n"
        "노드에 **마우스를 올리면** GNN 파급 확률 팝업이 표시됩니다."
    )

    top_k_val = top_k_edges if top_k_edges > 0 else 9999

    hier_net = build_hierarchical_network(
        path_nodes, path_edges,
        active_shocks, label_mapping, excluded_nodes,
        pure_adj, node_to_idx, tw_map, short_to_uri,
        probabilities,
        top_k=top_k_val,
        edge_threshold=edge_threshold,
        max_edge_width=max_edge_width,
        show_weak_edges=show_weak_edges
    )
    try:
        hier_net.save_graph("hier_graph.html")
        with open("hier_graph.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=560)
    except Exception as e:
        st.error(f"계층형 네트워크 렌더링 오류: {e}")

# ── TAB 3: 전체 원본 pyvis ──
with tab_full:
    st.markdown(
        "마우스 휠로 **확대/축소**, 노드 **드래그** 가능. "
        "$T_w$ 가중치가 엣지 두께에 반영됩니다. "
        "*(전체 그래프 — 복잡하지만 모든 관계 확인 가능)*"
    )

    net_full = Network(height="560px", width="100%", bgcolor="#1a1a2e",
                       font_color="#ecf0f1", directed=True)

    all_display_weights = []
    for s_n, p_n, o_n in raw_edges:
        if s_n not in node_to_idx or o_n not in node_to_idx: continue
        if s_n in excluded_nodes or o_n in excluded_nodes: continue
        si, oi = node_to_idx[s_n], node_to_idx[o_n]
        w = pure_adj[si, oi]
        if w > 0:
            all_display_weights.append(w)

    w_ref = np.percentile(all_display_weights, 95) if all_display_weights else 1.0

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
            title  = f"<b>{clean}</b><br>[이벤트 발원지]"
        elif node_name in excluded_nodes:
            color, size = "#4a4a6a", 14
            label  = f"⚫ {clean}"
            border = "#666688"
            title  = f"<b>{clean}</b><br>분석 제외"
        else:
            rr = probabilities[idx][2].item()
            yr = probabilities[idx][1].item()
            pg = probabilities[idx][0].item() * 100
            py = yr * 100
            pr = rr * 100
            if rr > 0.45:
                color, size, border = "#c0392b", 30, "#e74c3c"
            elif rr > 0.25:
                color, size, border = "#e67e22", 25, "#f39c12"
            elif yr > 0.4:
                color, size, border = "#f1c40f", 22, "#f39c12"
            else:
                color, size, border = "#27ae60", 18, "#2ecc71"
            label = clean
            # ← [NEW] hover 팝업에 확률 정보 포함
            title = (
                f"<b>{clean}</b><br>"
                f"🟢 영향 미미: {pg:.1f}%<br>"
                f"🟡 주의 필요: {py:.1f}%<br>"
                f"🔴 심각한 영향: {pr:.1f}%"
            )

        net_full.add_node(
            node_name, label=label,
            color={"background": color, "border": border},
            size=size, shape="dot",
            font={"size": 10, "color": "#ecf0f1"},
            title=title
        )

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

        normalized = display_w / w_ref
        scaled = np.log1p(normalized * 9) / np.log1p(9)
        e_width = max(float(scaled * max_edge_width), 0.5)

        is_shock_edge = (s_n in active_shocks) or \
                        (s_n == "Mat_Lithium" and "GPR_Channel" in active_shocks)
        if is_shock_edge:
            e_color = {"color": "#e74c3c", "opacity": 0.9}
        elif display_w > w_ref * 0.5:
            e_color = {"color": "#e67e22", "opacity": 0.7}
        else:
            e_color = {"color": "#5d6d7e", "opacity": 0.4}

        net_full.add_edge(
            s_n, o_n, value=e_width, color=e_color,
            title=(
                f"관계: {p_n}\n"
                f"BoK: {raw_w:.4f} | Tw: {tw_v:.4f}\n"
                f"최종 강도: {display_w:.4f}"
            ),
            arrows={"to": {"enabled": True, "scaleFactor": 0.5}}
        )

    net_full.toggle_physics(True)
    net_full.set_options("""
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
  "edges": { "smooth": {"type": "dynamic"} }
}
""")
    try:
        net_full.save_graph("pyvis_graph.html")
        with open("pyvis_graph.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=580)
    except Exception as e:
        st.error(f"네트워크 렌더링 오류: {e}")


# ══════════════════════════════════════════════
# [Frontend] Gemini 브리핑 + PDF 다운로드
# ══════════════════════════════════════════════
st.divider()
st.subheader("💡 이벤트 파급 영향 AI 브리핑")

if "briefing_text" not in st.session_state:
    st.session_state.briefing_text = None
if "last_shocks" not in st.session_state:
    st.session_state.last_shocks = []
if "briefing_loading" not in st.session_state:
    st.session_state.briefing_loading = False

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

        if st.session_state.briefing_text:
            if st.session_state.briefing_text.startswith("⚠️"):
                st.error(st.session_state.briefing_text)
            else:
                st.markdown(st.session_state.briefing_text)
        else:
            st.caption("위 버튼을 누르면 실시간 검색 기반 AI 브리핑이 생성됩니다.")
    else:
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
