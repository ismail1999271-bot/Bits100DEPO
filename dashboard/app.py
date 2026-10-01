"""Bits100 BIST research dashboard (Streamlit). Read-only: renders a snapshot JSON.

Run:  streamlit run dashboard/app.py -- --snapshot artifacts/dashboard/snapshot.json
No order buttons exist by design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

STATUS_COLOR = {"CONNECTED": "🟢", "STALE": "🟡", "INVALID": "🔴", "MISSING": "⚪", "BLOCKED": "⛔"}


def _snapshot_path() -> Path:
    args = sys.argv[1:]
    if "--snapshot" in args:
        return Path(args[args.index("--snapshot") + 1])
    import os

    return Path(os.environ.get("BITS100_SNAPSHOT", "artifacts/dashboard/snapshot.json"))


def _table(section: dict, title: str) -> None:
    st.subheader(title)
    if section.get("status") != "OK" or not section.get("data"):
        st.info(f"{section.get('status', 'MISSING')}: {section.get('reason', 'veri yok')}")
        return
    data = section["data"]
    st.dataframe(pd.DataFrame(data if isinstance(data, list) else [data]), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Bits100 BIST Research", layout="wide")
    path = _snapshot_path()
    st.title("Bits100 — BIST100+ Araştırma Paneli")
    if not path.exists():
        st.error(f"BLOCKED: snapshot bulunamadı ({path}). `python scripts/build_dashboard_snapshot.py` çalıştırın.")
        return
    snap = json.loads(path.read_text(encoding="utf-8"))
    st.caption(snap["disclaimer"] + f" · Oluşturma: {snap['generated_at']}")

    tab_main, tab_symbol, tab_plan = st.tabs(["Ana ekran", "Hisse detayı", "Plan / Risk"])
    with tab_main:
        c1, c2, c3 = st.columns(3)
        ms = snap["market_status"]
        c1.metric("Market Status", ms["phase"], ms["local_time"])
        uni = snap["universe"]
        c2.metric("BIST100+ Universe", uni["data"]["size"] if uni["status"] == "OK" else "BLOCKED")
        dq = snap["data_quality"]
        c3.metric("Data Quality (blocked)", dq["data"]["blocked"] if dq["status"] == "OK" else "—")
        st.caption(ms["note"])

        st.subheader("Provider Status")
        ps = snap["provider_status"]
        if ps["status"] == "OK":
            for row in ps["data"]:
                st.write(f"{STATUS_COLOR.get(row['status'], '')} **{row['label']}** — {row['status']} · {row['reason']}")
        else:
            st.warning(ps["reason"])

        _table(snap["top_quant_scores"], "Top Quant Scores")
        col_a, col_b = st.columns(2)
        with col_a:
            _table(snap["auction_leaders"], "Auction Leaders (09:40→09:55)")
            _table(snap["volume_leaders"], "Volume Leaders")
            _table(snap["kap_news"], "KAP / News")
        with col_b:
            _table(snap["tavan_dna_leaders"], "Tavan-DNA Leaders")
            _table(snap["institutional_flow"], "Institutional Flow")
            _table(snap["risk"], "Risk")
        paper = snap["paper_trading"]
        st.subheader("Paper Trading (simülasyon)")
        if paper["status"] == "OK":
            st.json(paper["data"]["summary"])
            if paper["data"].get("entries"):
                st.dataframe(pd.DataFrame(paper["data"]["entries"]), use_container_width=True)
        else:
            st.info(paper["reason"])

    with tab_plan:
        plan = snap.get("daily_plan", {"status": "MISSING", "reason": "no daily plan"})
        if plan["status"] == "OK":
            d = plan["data"]
            st.subheader(f"Günlük Plan {d['date']}")
            for item in d["items"]:
                st.write(f"{'✅' if item['done'] else '⬜'} {item['at']} — {item['text']}")
            st.write("İzleme listesi:", ", ".join(d["watchlist"]) or "YOK")
            if d["blocked"]:
                st.write("Engelli:", ", ".join(d["blocked"]))
        else:
            st.info(plan["reason"])
        _table(snap.get("stress_test", {"status": "MISSING", "reason": "no stress test"}), "Portföy Stres Testi")
        ov = snap.get("overfit_check", {"status": "MISSING", "reason": "no overfit check"})
        st.subheader("Aşırı Uyum Kontrolü")
        st.json(ov["data"]) if ov["status"] == "OK" else st.info(ov["reason"])

    with tab_symbol:
        symbols = sorted(snap.get("symbols", {}))
        if not symbols:
            st.info("Hisse detayı yok (gerçek veri bağlanmadı).")
            return
        choice = st.selectbox("Hisse", symbols)
        detail = snap["symbols"][choice]
        chart = detail.get("Chart")
        if chart and chart.get("status") == "OK":
            st.subheader("Grafik analizi")
            img = chart.get("image")
            if img and Path(img).exists():
                st.image(img, use_container_width=True)
            st.write("Destek:", chart["supports"] or "—", "· Direnç:", chart["resistances"] or "—")
            st.write("Formasyonlar:", ", ".join(p["label"] for p in chart["patterns"]) or "belirgin yok")
            st.caption("Formasyonlar geçmişi tanımlar; tahmin veya tavsiye değildir.")
        elif chart:
            st.info(chart.get("reason", "grafik yok"))
        for key in ("Price", "Auction", "Order Book", "Technical", "Tavan-DNA", "KAP", "News", "Fund",
                    "Institutional", "Broker", "Quant Score", "Backtest", "Risk"):
            with st.expander(key, expanded=key in ("Quant Score", "Auction")):
                value = detail.get(key)
                st.write("MISSING" if value is None else value)


if __name__ == "__main__":
    main()
