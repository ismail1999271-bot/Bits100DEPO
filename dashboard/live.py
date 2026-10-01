"""Bits100 - Canlı BIST Takip Ekranı (tüm hisseler, yoğun analiz).

Çalıştır:  streamlit run dashboard/live.py
Veri: BIST_DATA_BACKEND=borsapy (15 dk gecikmeli) veya lisanslı Level-1 sağlayıcı.
Kaynak yoksa ekran BLOCKED gösterir; örnek/sahte veri YOKTUR. Emir gönderme yoktur.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bist_hunter import live_feed  # noqa: E402
from bist_hunter.auction_features import TZ  # noqa: E402
from bist_hunter.chart_analysis import chart_report, render_chart  # noqa: E402
from bist_hunter.dashboard_data import market_status  # noqa: E402
from bist_hunter.live_board import change_histogram, movers  # noqa: E402

DISCLAIMER = "Araştırma/izleme ekranıdır; yatırım tavsiyesi değildir. Otomatik emir yoktur."
BADGE = {"OK": "🟢", "PARTIAL": "🟡", "BLOCKED": "⛔"}
COLS_MAIN = ["Sembol", "Son", "Değişim %", "Gün İçi Konum", "Hacim", "Değer (TL)", "Rel. Hacim", "RSI", "Trend",
             "Tavana Uzaklık %", "Tavan", "Taban", "Direnç Kırılımı", "ATR %", "Gap %", "Durum"]


@st.cache_data(ttl=3600, show_spinner="Günlük geçmiş yükleniyor (indikatörler için)…")
def _history_cached():
    return live_feed.load_history_for_env()


@st.cache_data(ttl=20, show_spinner=False)
def _snapshot_cached(_bucket: int):
    bars = _history_cached()
    return live_feed.load_live_snapshot(bars=bars, fetch_history=False)


def _config(df: pd.DataFrame) -> dict:
    cfg = {
        "Son": st.column_config.NumberColumn(format="%.2f"),
        "Değişim %": st.column_config.NumberColumn(format="%+.2f%%"),
        "Gün İçi Konum": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f",
                                                         help="0 = günün dibi, 1 = günün tepesi"),
        "Hacim": st.column_config.NumberColumn(format="%.0f"),
        "Değer (TL)": st.column_config.NumberColumn(format="%.0f"),
        "Rel. Hacim": st.column_config.NumberColumn(format="%.1fx", help="Bugünkü hacim / son 20 gün ortalaması"),
        "RSI": st.column_config.NumberColumn(format="%.0f"),
        "Tavana Uzaklık %": st.column_config.NumberColumn(format="%.2f%%"),
        "ATR %": st.column_config.NumberColumn(format="%.2f%%"),
        "Gap %": st.column_config.NumberColumn(format="%+.2f%%"),
    }
    return {k: v for k, v in cfg.items() if k in df.columns}


def _filter_widgets() -> dict:
    """Sidebar widgets live OUTSIDE the refreshing fragment; the fragment only applies the values."""
    sb = st.sidebar
    sb.header("Filtreler")
    return {
        "q": sb.text_input("Sembol ara", "").strip().upper(),
        "min_value": sb.number_input("Min. işlem değeri (milyon TL)", 0.0, 100000.0, 0.0, step=5.0),
        "direction": sb.radio("Yön", ["Hepsi", "Yükselen", "Düşen"], horizontal=True),
        "only": sb.multiselect("Sadece", ["Tavan", "Tavana yakın (≤%2)", "Taban", "Direnç kırılımı",
                                          "Rel. hacim ≥2x", "Trend yukarı"]),
    }


def _apply_filters(board: pd.DataFrame, f: dict) -> pd.DataFrame:
    out, only = board, f["only"]
    if f["q"]:
        out = out[out["Sembol"].str.contains(f["q"], regex=False)]
    if f["min_value"]:
        out = out[out["Değer (TL)"].fillna(0) >= f["min_value"] * 1e6]
    if f["direction"] == "Yükselen":
        out = out[out["Değişim %"] > 0]
    elif f["direction"] == "Düşen":
        out = out[out["Değişim %"] < 0]
    if "Tavan" in only:
        out = out[out["Tavan"]]
    if "Tavana yakın (≤%2)" in only:
        out = out[(out["Tavana Uzaklık %"] > 0) & (out["Tavana Uzaklık %"] <= 2)]
    if "Taban" in only:
        out = out[out["Taban"]]
    if "Direnç kırılımı" in only and "Direnç Kırılımı" in out:
        out = out[out["Direnç Kırılımı"] == True]  # noqa: E712
    if "Rel. hacim ≥2x" in only and "Rel. Hacim" in out:
        out = out[out["Rel. Hacim"] >= 2]
    if "Trend yukarı" in only and "Trend" in out:
        out = out[out["Trend"] == "YUKARI"]
    return out


def _table(df: pd.DataFrame, cols=None, height: int = 520) -> None:
    show = [c for c in (cols or COLS_MAIN) if c in df.columns]
    view = df[show]

    def tint(v):
        if pd.isna(v):
            return ""
        return "color: #16a34a" if v > 0 else "color: #dc2626" if v < 0 else ""

    styled = view.style.map(tint, subset=[c for c in ("Değişim %", "Gap %") if c in view.columns])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=height, column_config=_config(df))


def _kpis(s: dict, snap) -> None:
    c = st.columns(6)
    c[0].metric("Hisse", s["count"], f"{len(snap.skipped)} atlandı" if snap.skipped else None, delta_color="off")
    c[1].metric("Yükselen / Düşen", f"{s['yükselen']} / {s['düşen']}", f"genişlik %{s['genişlik %']}")
    c[2].metric("Tavan / Taban", f"{s['tavan']} / {s['taban']}")
    c[3].metric("Medyan değişim", f"%{s['medyan değişim %']:+.2f}")
    c[4].metric("Toplam değer", f"{s['toplam değer (TL)'] / 1e9:,.1f} mlr TL")
    c[5].metric("STALE", s["stale"], delta_color="off")


def _scatter(df: pd.DataFrame) -> None:
    import altair as alt

    d = df.dropna(subset=["Değişim %", "Değer (TL)"]).copy()
    if "Rel. Hacim" not in d.columns:
        st.info("Rel. hacim için günlük geçmiş verisi yok; yalnızca değişim dağılımı gösterilir.")
        return
    d = d.dropna(subset=["Rel. Hacim"])
    # Vega-Lite treats "." in field names as nesting, so chart on plain ASCII field names
    c = pd.DataFrame({"sembol": d["Sembol"], "son": d["Son"], "degisim": d["Değişim %"],
                      "relhacim": d["Rel. Hacim"].clip(lower=0.1), "deger": d["Değer (TL)"],
                      "rsi": d["RSI"] if "RSI" in d else None, "trend": d["Trend"] if "Trend" in d else None})
    c["yon"] = ["Yükselen" if x >= 0 else "Düşen" for x in c["degisim"]]
    chart = alt.Chart(c).mark_circle(opacity=0.7).encode(
        x=alt.X("relhacim:Q", scale=alt.Scale(type="log"), title="Göreli hacim (x, log ölçek)"),
        y=alt.Y("degisim:Q", title="Günlük değişim %"),
        size=alt.Size("deger:Q", legend=None, scale=alt.Scale(range=[30, 700])),
        color=alt.Color("yon:N", scale=alt.Scale(domain=["Yükselen", "Düşen"], range=["#16a34a", "#dc2626"]),
                        legend=None),
        tooltip=[alt.Tooltip("sembol:N", title="Sembol"), alt.Tooltip("son:Q", title="Son"),
                 alt.Tooltip("degisim:Q", title="Değişim %", format="+.2f"),
                 alt.Tooltip("relhacim:Q", title="Rel. hacim", format=".1f"),
                 alt.Tooltip("deger:Q", title="Değer (TL)", format=",.0f"),
                 alt.Tooltip("rsi:Q", title="RSI", format=".0f"), alt.Tooltip("trend:N", title="Trend")]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption("Sağ üst: hacimle yükselen · sağ alt: hacimle düşen. Baloncuk büyüklüğü = işlem değeri.")


def _alert_log(snap) -> None:
    log = st.session_state.setdefault("alert_log", [])
    seen = st.session_state.setdefault("alert_seen", set())
    stamp = snap.fetched_at.astimezone(TZ).strftime("%H:%M:%S")
    for a in snap.alerts:
        if a["id"] not in seen:
            seen.add(a["id"])
            log.insert(0, {"Saat": stamp, "Tür": a["kind"], "Uyarı": a["text"]})
    del log[500:]
    st.subheader("Oturum uyarıları (yeni olaylar)")
    if log:
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True, height=420)
    else:
        st.info("Henüz yeni olay yok. Aynı olay tekrar tekrar listelenmez.")


def _detail(snap, board: pd.DataFrame) -> None:
    sym = st.selectbox("Hisse", list(board["Sembol"]))
    row = board[board["Sembol"] == sym].iloc[0]
    c = st.columns(5)
    c[0].metric("Son", f"{row['Son']:.2f}", f"{row['Değişim %']:+.2f}%")
    c[1].metric("Tavan fiyatı", f"{row['Tavan Fiyatı']:.2f}", f"%{row['Tavana Uzaklık %']:.2f} uzak", delta_color="off")
    c[2].metric("Taban fiyatı", f"{row['Taban Fiyatı']:.2f}")
    c[3].metric("RSI", "—" if pd.isna(row.get("RSI")) else f"{row['RSI']:.0f}")
    c[4].metric("Rel. hacim", "—" if pd.isna(row.get("Rel. Hacim")) else f"{row['Rel. Hacim']:.1f}x")
    if snap.bars is None or sym not in set(snap.bars["symbol"]):
        st.info("Bu hisse için günlük geçmiş yok: grafik MISSING.")
        return
    rep = chart_report(snap.bars, sym)
    if rep["status"] != "OK":
        st.info(rep["reason"])
        return
    out = render_chart(snap.bars, sym, Path("artifacts/live_charts") / f"{sym}.png")
    if out:
        st.image(str(out), use_container_width=True)
    st.write("Destek:", rep["supports"] or "—", "· Direnç:", rep["resistances"] or "—")
    st.write("Formasyonlar:", ", ".join(p["label"] for p in rep["patterns"]) or "belirgin yok")
    st.caption("Grafik son kapanmış günlük mumlara dayanır; formasyonlar geçmişi tanımlar, tahmin değildir.")


def _body(interval: int, filters: dict) -> None:
    now = datetime.now(timezone.utc)
    snap = _snapshot_cached(int(now.timestamp() // max(interval, 20)))
    local = snap.fetched_at.astimezone(TZ)
    ms = market_status(now)
    st.markdown(f"{BADGE.get(snap.status, '')} **{snap.status}** · {snap.reason} · piyasa: **{ms['phase']}** · "
                f"son çekim {local.strftime('%H:%M:%S')} · kaynak: {snap.source}")
    if snap.note:
        st.warning(snap.note)
    if snap.status == "BLOCKED" or snap.board is None:
        st.error("Canlı veri yok. " + snap.reason)
        st.code("BIST_DATA_BACKEND=borsapy   # 15 dk gecikmeli, kişisel kullanım\n"
                "pip install borsapy\nstreamlit run dashboard/live.py")
        return
    _kpis(snap.summary, snap)
    board = _apply_filters(snap.board, filters)
    st.caption(f"{len(board)} / {len(snap.board)} hisse filtreye uyuyor")
    t = st.tabs(["Piyasa Haritası", "Tüm Hisseler", "Hareketliler", "Tavan / Taban", "Hisse Analizi", "Uyarılar"])
    with t[0]:
        a, b = st.columns([1, 2])
        with a:
            st.subheader("Değişim dağılımı")
            h = change_histogram(snap.board)
            import altair as alt

            h["Renk"] = ["#dc2626"] * 5 + ["#16a34a"] * 5
            st.altair_chart(alt.Chart(h).mark_bar().encode(
                x=alt.X("Aralık:N", sort=list(h["Aralık"]), title="Günlük değişim % aralığı"),
                y=alt.Y("Adet:Q", title="Hisse sayısı"), color=alt.Color("Renk:N", scale=None),
                tooltip=["Aralık", "Adet"]), use_container_width=True)
        with b:
            st.subheader("Hacim × Değişim")
            _scatter(board)
    with t[1]:
        _table(board)
    with t[2]:
        a, b = st.columns(2)
        cols = ["Sembol", "Son", "Değişim %", "Değer (TL)", "Rel. Hacim"]
        with a:
            st.subheader("En çok yükselen")
            _table(movers(board, "Değişim %"), cols, 380)
            st.subheader("En yüksek işlem değeri")
            _table(movers(board, "Değer (TL)"), cols, 380)
        with b:
            st.subheader("En çok düşen")
            _table(movers(board, "Değişim %", ascending=True), cols, 380)
            st.subheader("En yüksek göreli hacim")
            _table(movers(board, "Rel. Hacim"), cols, 380)
    with t[3]:
        a, b = st.columns(2)
        with a:
            st.subheader("Tavanda")
            _table(board[board["Tavan"]], height=260)
            st.subheader("Tavana yakın (≤%2)")
            near = board[(board["Tavana Uzaklık %"] > 0) & (board["Tavana Uzaklık %"] <= 2)]
            _table(near.sort_values("Tavana Uzaklık %"), height=300)
        with b:
            st.subheader("Tabanda")
            _table(board[board["Taban"]], height=260)
            if "Direnç Kırılımı" in board:
                st.subheader("20g direnç kırılımı")
                _table(board[board["Direnç Kırılımı"] == True], height=300)  # noqa: E712
        st.caption("Tavan/taban ±%10 limite göre hesaplanır; VBTS/özel limitler ve seans durumu ayrıca kontrol edilmelidir.")
    with t[4]:
        if board.empty:
            st.info("Filtreye uyan hisse yok.")
        else:
            _detail(snap, board)
    with t[5]:
        _alert_log(snap)


def main() -> None:
    st.set_page_config(page_title="Bits100 Canlı BIST", layout="wide")
    st.title("Bits100 — Canlı BIST Takip")
    st.caption(DISCLAIMER)
    interval = st.sidebar.select_slider("Yenileme (sn)", [30, 60, 120, 300], value=60)
    st.sidebar.caption("Veri gecikmesi kaynağa bağlıdır; ekranda yaş ve STALE işaretlenir.")
    filters = _filter_widgets()
    # run_every is fixed per fragment definition, so build it with the chosen period
    st.fragment(run_every=interval)(_body)(interval, filters)


main()
