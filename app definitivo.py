"""
App Streamlit — CBLOL Rating System (Elo + Glicko-2)
========================================================
Dados já embutidos no código (embedded_data.py) — não precisa
fazer upload de CSV nenhum, o app já sobe pronto com dados de
2024-2026 do CBLOL.

Rodar localmente:
    pip install streamlit pandas numpy plotly
    streamlit run app.py

Pra atualizar os dados no futuro, gere um novo embedded_data.py
(veja o script build_embedded_data.py) e substitua o arquivo.
"""

import io
import gzip
import base64
import math

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from embedded_data import CBLOL_DATA_B85

st.set_page_config(page_title="CBLOL Rating System", page_icon="🎮", layout="wide")

PURPLE = "#8C52FF"
GOLD = "#F2C230"
DARK = "#0E0E16"

st.markdown(f"""
<style>
    .stApp {{ background-color: {DARK}; }}
    [data-testid="stMetricValue"] {{ color: {GOLD}; }}
    h1, h2, h3 {{ color: {PURPLE}; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 16px; padding: 8px 16px; }}
    div[data-testid="stMetric"] {{
        background-color: rgba(140, 82, 255, 0.08);
        border: 1px solid rgba(140, 82, 255, 0.25);
        border-radius: 12px;
        padding: 12px;
    }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# CARREGAMENTO DOS DADOS EMBUTIDOS
# ----------------------------------------------------------------------

@st.cache_data
def load_embedded_csv():
    compressed = base64.b85decode(CBLOL_DATA_B85)
    raw = gzip.decompress(compressed)
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


@st.cache_data
def build_matches(df, leagues):
    d = df.copy()
    if "position" in d.columns:
        d = d[d["position"].str.lower() == "team"]
    d = d[d["league"].isin(leagues)].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    matches = []
    for gid, g in d.groupby("gameid"):
        if len(g) != 2:
            continue
        row_blue = g[g["side"].str.lower() == "blue"]
        row_red = g[g["side"].str.lower() == "red"]
        if row_blue.empty or row_red.empty:
            continue
        row_blue, row_red = row_blue.iloc[0], row_red.iloc[0]
        matches.append({
            "date": row_blue["date"],
            "team_blue": row_blue["teamname"],
            "team_red": row_red["teamname"],
            "blue_won": int(row_blue["result"]),
        })
    return pd.DataFrame(matches).sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------
# RATING SYSTEMS
# ----------------------------------------------------------------------

class EloRatingSystem:
    def __init__(self, k=24, base_rating=1500):
        self.k = k
        self.base_rating = base_rating
        self.ratings = {}

    def get(self, team):
        return self.ratings.get(team, self.base_rating)

    def win_probability(self, team_a, team_b):
        ra, rb = self.get(team_a), self.get(team_b)
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, team_a, team_b, a_won: bool):
        pa = self.win_probability(team_a, team_b)
        score_a = 1.0 if a_won else 0.0
        ra, rb = self.get(team_a), self.get(team_b)
        self.ratings[team_a] = ra + self.k * (score_a - pa)
        self.ratings[team_b] = rb + self.k * ((1 - score_a) - (1 - pa))


class Glicko2Team:
    def __init__(self, rating=1500, rd=350, vol=0.06):
        self.rating = rating
        self.rd = rd
        self.vol = vol


class Glicko2RatingSystem:
    TAU = 0.5
    SCALE = 173.7178

    def __init__(self):
        self.teams = {}

    def _get(self, name):
        if name not in self.teams:
            self.teams[name] = Glicko2Team()
        return self.teams[name]

    def _to_scale(self, team):
        return (team.rating - 1500) / self.SCALE, team.rd / self.SCALE

    def win_probability(self, team_a, team_b):
        ta, tb = self._get(team_a), self._get(team_b)
        mu_a, phi_a = self._to_scale(ta)
        mu_b, phi_b = self._to_scale(tb)
        g_phi = 1 / math.sqrt(1 + 3 * phi_b ** 2 / math.pi ** 2)
        return 1 / (1 + math.exp(-g_phi * (mu_a - mu_b)))

    def update(self, team_a, team_b, a_won: bool):
        self._update_one(team_a, team_b, 1.0 if a_won else 0.0)
        self._update_one(team_b, team_a, 0.0 if a_won else 1.0)

    def _update_one(self, name, opp_name, score):
        team, opp = self._get(name), self._get(opp_name)
        mu, phi = self._to_scale(team)
        mu_j, phi_j = self._to_scale(opp)
        g_j = 1 / math.sqrt(1 + 3 * phi_j ** 2 / math.pi ** 2)
        E_j = 1 / (1 + math.exp(-g_j * (mu - mu_j)))
        v = 1 / (g_j ** 2 * E_j * (1 - E_j) + 1e-10)
        delta = v * g_j * (score - E_j)
        a = math.log(team.vol ** 2)
        A = a
        eps = 1e-6
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while self._f(a - k * self.TAU, delta, phi, v, a) < 0:
                k += 1
            B = a - k * self.TAU
        fA, fB = self._f(A, delta, phi, v, a), self._f(B, delta, phi, v, a)
        while abs(B - A) > eps:
            C = A + (A - B) * fA / (fB - fA)
            fC = self._f(C, delta, phi, v, a)
            if fC * fB < 0:
                A, fA = B, fB
            else:
                fA /= 2
            B, fB = C, fC
        new_vol = math.exp(A / 2)
        phi_star = math.sqrt(phi ** 2 + new_vol ** 2)
        new_phi = 1 / math.sqrt(1 / phi_star ** 2 + 1 / v)
        new_mu = mu + new_phi ** 2 * g_j * (score - E_j)
        team.rating = new_mu * self.SCALE + 1500
        team.rd = new_phi * self.SCALE
        team.vol = new_vol

    def _f(self, x, delta, phi, v, a):
        ex = math.exp(x)
        num = ex * (delta ** 2 - phi ** 2 - v - ex)
        den = 2 * (phi ** 2 + v + ex) ** 2
        return (num / den) - (x - a) / self.TAU ** 2


@st.cache_data
def run_ratings(matches):
    elo = EloRatingSystem()
    g2 = Glicko2RatingSystem()
    elo_history, g2_history = [], []
    elo_preds, g2_preds = [], []

    for _, row in matches.iterrows():
        elo_preds.append(elo.win_probability(row["team_blue"], row["team_red"]))
        g2_preds.append(g2.win_probability(row["team_blue"], row["team_red"]))

        elo.update(row["team_blue"], row["team_red"], bool(row["blue_won"]))
        g2.update(row["team_blue"], row["team_red"], bool(row["blue_won"]))

        for team in (row["team_blue"], row["team_red"]):
            elo_history.append({"date": row["date"], "team": team, "rating": elo.get(team)})
            g2_history.append({"date": row["date"], "team": team, "rating": g2._get(team).rating})

    return elo, g2, pd.DataFrame(elo_history), pd.DataFrame(g2_history), np.array(elo_preds), np.array(g2_preds)


def metrics(preds, actual):
    eps = 1e-10
    logloss = -np.mean(actual * np.log(preds + eps) + (1 - actual) * np.log(1 - preds + eps))
    acc = np.mean((preds > 0.5).astype(int) == actual)
    return logloss, acc


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("🎮 CBLOL Rating System")
st.caption("Sistema de rating Elo + Glicko-2 para o CBLOL, com backtesting sobre dados reais (2024–2026) do Oracle's Elixir")

raw_df = load_embedded_csv()
matches = build_matches(raw_df, ("CBLOL", "LTA S"))

st.sidebar.header("🎮 CBLOL Rating System")
st.sidebar.caption(f"{len(matches)} partidas carregadas · {matches['date'].dt.year.min()}–{matches['date'].dt.year.max()}")
st.sidebar.divider()
with st.sidebar.expander("ℹ️ Sobre os dados"):
    st.write(
        "Base construída a partir do Oracle's Elixir. A partir de 2025 a liga "
        "passou a se chamar **LTA South**; os jogos foram filtrados para manter "
        "apenas confrontos entre organizações brasileiras, equivalendo ao CBLOL histórico."
    )

elo, g2, elo_hist, g2_hist, elo_preds, g2_preds = run_ratings(matches)
actual = matches["blue_won"].values
elo_ll, elo_acc = metrics(elo_preds, actual)
g2_ll, g2_acc = metrics(g2_preds, actual)
baseline_acc = max(actual.mean(), 1 - actual.mean())

tab1, tab2, tab3 = st.tabs(["📊 Métricas", "📈 Evolução de rating", "⚔️ Simular confronto"])

with tab1:
    st.subheader("Desempenho do modelo (backtesting walk-forward)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia — Baseline", f"{baseline_acc:.1%}")
    col2.metric("Acurácia — Elo", f"{elo_acc:.1%}", f"{(elo_acc - baseline_acc):+.1%}")
    col3.metric("Acurácia — Glicko-2", f"{g2_acc:.1%}", f"{(g2_acc - baseline_acc):+.1%}")
    st.caption(f"Log loss — Elo: {elo_ll:.4f} | Glicko-2: {g2_ll:.4f} (quanto menor, melhor)")

    st.divider()
    st.subheader("🏆 Ranking atual dos times (Elo)")
    ranking = pd.DataFrame(
        [(t, r) for t, r in elo.ratings.items()], columns=["Time", "Rating Elo"]
    ).sort_values("Rating Elo", ascending=False).reset_index(drop=True)
    ranking.index += 1
    ranking["Rating Elo"] = ranking["Rating Elo"].round(1)
    st.dataframe(ranking, use_container_width=True)

with tab2:
    st.subheader("Evolução do rating ao longo do tempo")
    system_choice = st.radio("Sistema", ["Elo", "Glicko-2"], horizontal=True)
    hist = elo_hist if system_choice == "Elo" else g2_hist
    all_teams = sorted(hist["team"].unique())
    teams = st.multiselect("Times pra mostrar", options=all_teams, default=all_teams[:5])
    fig = go.Figure()
    for team in teams:
        team_data = hist[hist["team"] == team]
        fig.add_trace(go.Scatter(x=team_data["date"], y=team_data["rating"],
                                  mode="lines+markers", name=team))
    fig.update_layout(
        xaxis_title="Data", yaxis_title="Rating", height=520,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Calculadora de confronto")
    teams_list = sorted(elo.ratings.keys())
    c1, c2 = st.columns(2)
    team_a = c1.selectbox("Time (lado azul)", teams_list, index=0)
    team_b = c2.selectbox("Time (lado vermelho)", teams_list, index=min(1, len(teams_list) - 1))

    if team_a == team_b:
        st.warning("Escolha dois times diferentes.")
    else:
        p_elo = elo.win_probability(team_a, team_b)
        p_g2 = g2.win_probability(team_a, team_b)
        c1.metric(f"{team_a} vence (Elo)", f"{p_elo:.1%}")
        c1.metric(f"{team_a} vence (Glicko-2)", f"{p_g2:.1%}")
        c2.metric(f"{team_b} vence (Elo)", f"{1 - p_elo:.1%}")
        c2.metric(f"{team_b} vence (Glicko-2)", f"{1 - p_g2:.1%}")

st.divider()
st.caption("Backtesting walk-forward: cada previsão usa só dados anteriores à partida, sem olhar o futuro.")
