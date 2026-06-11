from __future__ import annotations
import io
import numpy as np
import pandas as pd


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in sheets.items():
            safe = name[:31]
            df.to_excel(writer, index=False, sheet_name=safe)
    return output.getvalue()


def _fmt(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
        return f"{float(v):,.0f}".replace(',', ' ')
    return str(v)


def _small_table(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 5):
    from reportlab.platypus import Table
    from reportlab.lib import colors
    if df is None or len(df) == 0:
        data = [['No data']]
    else:
        d = df.copy().head(max_rows).iloc[:, :max_cols]
        data = [list(d.columns)] + [[_fmt(x) for x in row] for row in d.to_numpy()]
    tbl = Table(data, repeatRows=1)
    tbl.setStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0B1F3A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('GRID', (0,0), (-1,-1), .25, colors.HexColor('#D0D5DD')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ])
    return tbl


def _bar_chart(values: dict, title: str, width=430, height=180):
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.lib import colors
    d = Drawing(width, height + 25)
    d.add(String(0, height + 8, title, fontSize=10, fillColor=colors.HexColor('#0B1F3A')))
    chart = VerticalBarChart()
    chart.x = 30
    chart.y = 35
    chart.height = height - 45
    chart.width = width - 60
    vals = list(values.values()) or [0]
    labs = list(values.keys()) or ['-']
    chart.data = [vals]
    chart.categoryAxis.categoryNames = labs
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.fontSize = 6
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = colors.HexColor('#1F77B4')
    d.add(chart)
    return d


def _hist_chart(arr, title: str, bins: int = 12, width=430, height=180):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        hist = np.array([0]); labels = ['-']
    else:
        hist, edges = np.histogram(arr, bins=bins)
        labels = [str(i+1) for i in range(len(hist))]
    return _bar_chart(dict(zip(labels, hist)), title, width, height)


def build_pdf_report(payload: dict) -> bytes:
    """Builds a lightweight client PDF report without requiring Chrome/Kaleido.

    The report contains KPIs, method comparisons, risk tables and
    synthetic charts created with ReportLab.
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.3*cm, leftMargin=1.3*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='NavyTitle', parent=styles['Title'], textColor=colors.HexColor('#0B1F3A'), fontSize=18, leading=22))
    styles.add(ParagraphStyle(name='NavyH2', parent=styles['Heading2'], textColor=colors.HexColor('#0B1F3A'), fontSize=13, leading=16))
    styles.add(ParagraphStyle(name='SmallMuted', parent=styles['BodyText'], textColor=colors.HexColor('#667085'), fontSize=8, leading=11))

    story = []
    title = payload.get('title', 'Individual reserving report')
    story.append(Paragraph(title, styles['NavyTitle']))
    story.append(Paragraph(f"Valuation year: {payload.get('valuation_year', '-')}", styles['SmallMuted']))
    meta = payload.get('metadata', {})
    if isinstance(meta, dict) and meta:
        meta_df = pd.DataFrame({'Item': list(meta.keys()), 'Value': list(meta.values())})
        story.append(Spacer(1, 6))
        story.append(_small_table(meta_df, max_rows=10, max_cols=2))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Executive summary", styles['NavyH2']))
    story.append(Paragraph(
        "This report summarises the results of MarkovReserve, a semi-Markov individual claims reserving engine: KPIs, reserves, CDR, transition diagnostics, stress indicators and comparison with collective benchmarks.",
        styles['BodyText']))
    exec_df = payload.get('executive_summary', pd.DataFrame())
    if isinstance(exec_df, pd.DataFrame) and len(exec_df):
        story.append(Spacer(1, 6))
        story.append(_small_table(exec_df, max_rows=8, max_cols=1))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Key performance indicators", styles['NavyH2']))
    story.append(_small_table(payload.get('kpis', pd.DataFrame()), max_rows=12, max_cols=3))
    dq = payload.get('data_quality', pd.DataFrame())
    if isinstance(dq, pd.DataFrame) and len(dq):
        story.append(Spacer(1, 10))
        story.append(Paragraph("Data quality", styles['NavyH2']))
        story.append(_small_table(dq, max_rows=10, max_cols=2))
    econ = payload.get('economic_assumptions', pd.DataFrame())
    if isinstance(econ, pd.DataFrame) and len(econ):
        story.append(Spacer(1, 10))
        story.append(Paragraph("Economic assumptions", styles['NavyH2']))
        story.append(_small_table(econ, max_rows=10, max_cols=2))

    methodology = payload.get('methodology', pd.DataFrame())
    if isinstance(methodology, pd.DataFrame) and len(methodology):
        story.append(Spacer(1, 10))
        story.append(Paragraph("Methodology appendix", styles['NavyH2']))
        story.append(Paragraph("The table below summarises the main actuarial concepts used in MarkovReserve in non-technical language.", styles['BodyText']))
        story.append(Spacer(1, 6))
        story.append(_small_table(methodology, max_rows=12, max_cols=2))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Method comparison", styles['NavyH2']))
    comp = payload.get('comparison', pd.DataFrame())
    story.append(_small_table(comp, max_rows=10, max_cols=4))
    if isinstance(comp, pd.DataFrame) and len(comp) and 'Method' in comp and 'Reserve' in comp:
        story.append(Spacer(1, 8))
        story.append(_bar_chart(dict(zip(comp['Method'].astype(str), comp['Reserve'].astype(float))), "Reserves by method"))

    story.append(PageBreak())
    story.append(Paragraph("Total reserve and risk", styles['NavyH2']))
    story.append(_small_table(payload.get('risk_total_reserve', pd.DataFrame()), max_rows=12, max_cols=2))
    story.append(Spacer(1, 8))
    story.append(_hist_chart(payload.get('reserve_values', []), "Simulated total reserve distribution"))

    story.append(Spacer(1, 12))
    story.append(Paragraph("One-year CDR", styles['NavyH2']))
    story.append(_small_table(payload.get('risk_cdr', pd.DataFrame()), max_rows=12, max_cols=2))
    story.append(Spacer(1, 8))
    story.append(_hist_chart(payload.get('cdr_values', []), "Simulated CDR distribution"))

    story.append(PageBreak())
    story.append(Paragraph("States, transitions and payments", styles['NavyH2']))
    story.append(Paragraph("Snapshot by state", styles['Heading3']))
    story.append(_small_table(payload.get('state_snapshot', pd.DataFrame()), max_rows=12, max_cols=3))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Main transitions", styles['Heading3']))
    story.append(_small_table(payload.get('transitions', pd.DataFrame()), max_rows=15, max_cols=4))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Payment models", styles['Heading3']))
    story.append(_small_table(payload.get('payment_models', pd.DataFrame()), max_rows=15, max_cols=5))

    story.append(PageBreak())
    story.append(Paragraph("IBNR and collective benchmarks", styles['NavyH2']))
    story.append(Paragraph("IBNR by accident year", styles['Heading3']))
    story.append(_small_table(payload.get('ibnr', pd.DataFrame()), max_rows=15, max_cols=5))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Chain-Ladder factors", styles['Heading3']))
    story.append(_small_table(payload.get('cl_factors', pd.DataFrame()), max_rows=15, max_cols=3))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Benchmark detail by year", styles['Heading3']))
    story.append(_small_table(payload.get('bench_by_ay', pd.DataFrame()), max_rows=15, max_cols=6))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Note: this report is generated automatically. Assumptions, data quality and results must be validated by an actuary before any regulatory or decision-making use.", styles['SmallMuted']))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
