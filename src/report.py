"""
TerrainAI Tactical — PDF report generator.
Produces a 5-page professional mission briefing document.
"""

import base64
import io
import datetime
from pathlib import Path

from PIL import Image as PILImage
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)

# ─── Palette ─────────────────────────────────────────────────────────────────

COLOR_DARK      = colors.HexColor('#1a1a2e')
COLOR_ACCENT    = colors.HexColor('#e94560')
COLOR_AMBER     = colors.HexColor('#f59e0b')
COLOR_GREEN     = colors.HexColor('#10b981')
COLOR_RED       = colors.HexColor('#ef4444')
COLOR_BLUE      = colors.HexColor('#3b82f6')
COLOR_LIGHT_BG  = colors.HexColor('#f8fafc')
COLOR_MID_GRAY  = colors.HexColor('#64748b')
COLOR_BORDER    = colors.HexColor('#cbd5e1')
COLOR_AMBER_BG  = colors.HexColor('#fffbeb')
COLOR_RED_BG    = colors.HexColor('#fef2f2')
COLOR_GREEN_BG  = colors.HexColor('#f0fdf4')

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm

ABSENT_CLASS_IDS = {5, 6}

# Per-class IoU from the best run (N/A for absent classes)
CLASS_IOUs = {
    0: 0.4821,
    1: 0.5103,
    2: 0.5892,
    3: 0.6217,
    4: 0.5934,
    5: None,   # Trees — absent
    6: None,   # Water — absent
    7: 0.7841,
    8: 0.6504,
    9: 0.9801,
}

CLASS_NAMES = {
    0: 'Sand', 1: 'Gravel', 2: 'Rocks', 3: 'Dirt', 4: 'Grass',
    5: 'Trees', 6: 'Water', 7: 'Sky', 8: 'Logs', 9: 'Flowers',
}

TRAVERSABLE_IDS = {0, 1, 3, 4}


def _styles():
    base = getSampleStyleSheet()
    custom = {
        'cover_title': ParagraphStyle(
            'cover_title', parent=base['Normal'],
            fontSize=22, textColor=colors.white, fontName='Helvetica-Bold',
            alignment=TA_CENTER, spaceAfter=4,
        ),
        'cover_sub': ParagraphStyle(
            'cover_sub', parent=base['Normal'],
            fontSize=10, textColor=COLOR_ACCENT, fontName='Helvetica-Bold',
            alignment=TA_CENTER, spaceAfter=6,
        ),
        'cover_meta': ParagraphStyle(
            'cover_meta', parent=base['Normal'],
            fontSize=9, textColor=COLOR_MID_GRAY,
            alignment=TA_CENTER, spaceAfter=3,
        ),
        'section_heading': ParagraphStyle(
            'section_heading', parent=base['Normal'],
            fontSize=13, textColor=COLOR_DARK, fontName='Helvetica-Bold',
            spaceBefore=12, spaceAfter=6,
        ),
        'body': ParagraphStyle(
            'body', parent=base['Normal'],
            fontSize=9, textColor=colors.HexColor('#1e293b'),
            spaceAfter=4, leading=14,
        ),
        'score_big': ParagraphStyle(
            'score_big', parent=base['Normal'],
            fontSize=36, textColor=COLOR_DARK, fontName='Helvetica-Bold',
            alignment=TA_CENTER,
        ),
        'rating': ParagraphStyle(
            'rating', parent=base['Normal'],
            fontSize=14, fontName='Helvetica-Bold',
            alignment=TA_CENTER, spaceAfter=4,
        ),
        'alert_text': ParagraphStyle(
            'alert_text', parent=base['Normal'],
            fontSize=9, textColor=colors.white, fontName='Helvetica-Bold',
            spaceAfter=2,
        ),
        'footnote': ParagraphStyle(
            'footnote', parent=base['Normal'],
            fontSize=7.5, textColor=COLOR_MID_GRAY,
            spaceAfter=4, leading=11,
        ),
    }
    return custom


def _b64_to_rl_image(b64_str: str, max_width: float) -> RLImage:
    """Decode a base64 PNG/JPEG and return a ReportLab Image at max_width."""
    img_bytes = base64.b64decode(b64_str)
    pil_img = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')
    w, h = pil_img.size
    aspect = h / w
    display_w = max_width
    display_h = display_w * aspect
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return RLImage(buf, width=display_w, height=display_h)


def _action_color(action: str):
    action = action.upper()
    if 'ADVANCE' in action:
        return COLOR_GREEN, COLOR_GREEN_BG
    if 'CAUTION' in action:
        return COLOR_AMBER, COLOR_AMBER_BG
    return COLOR_RED, COLOR_RED_BG


def _threat_color(level: str):
    level = level.upper()
    if level == 'LOW':
        return COLOR_GREEN
    if level == 'MEDIUM':
        return COLOR_AMBER
    return COLOR_RED


def generate_pdf_report(session_data: dict, output_path: str) -> str:
    """
    Generate a 5-page PDF tactical briefing.
    session_data must contain keys produced by /analyze/image endpoint.
    Returns output_path.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title='TerrainAI Tactical Briefing',
    )

    S = _styles()
    story = []
    content_width = PAGE_W - 2 * MARGIN

    session_id  = session_data.get('session_id', 'N/A')
    filename    = session_data.get('filename', 'unknown')
    stats       = session_data.get('stats', {})
    analysis    = session_data.get('analysis', {})
    traversability = analysis.get('traversability', {})
    threat         = analysis.get('threat', {})
    recommendation = analysis.get('recommendation', {})
    timestamp   = session_data.get('analysis', {}).get('timestamp', datetime.datetime.utcnow().isoformat() + 'Z')

    per_class = stats.get('per_class', {})
    # Normalise keys to int (JSON loads them as strings)
    per_class = {int(k): v for k, v in per_class.items()}

    # ─── PAGE 1 — COVER ──────────────────────────────────────────────────────

    # Dark header bar via single-cell table
    header_data = [[Paragraph('TERRAIN INTELLIGENCE BRIEFING', S['cover_title'])]]
    header_table = Table(header_data, colWidths=[content_width])
    header_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, -1), COLOR_DARK),
        ('TOPPADDING',  (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph('EXERCISE USE ONLY — NOT FOR OPERATIONAL USE', S['cover_sub']))
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width=content_width, thickness=1, color=COLOR_BORDER))
    story.append(Spacer(1, 0.4 * cm))

    meta_rows = [
        ['Session ID',  session_id],
        ['Generated',   timestamp[:19].replace('T', ' ') + ' UTC'],
        ['Source File', filename],
        ['Model',       'SegFormer-B2  |  mIoU: 0.6109  |  Flowers IoU: 0.9801  |  Logs IoU: 0.6504'],
        ['Classes',     '10 defined  |  8 present in dataset  |  2 absent (Trees, Water)'],
    ]
    meta_table = Table(meta_rows, colWidths=[4 * cm, content_width - 4 * cm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME',    (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',    (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE',    (0, 0), (-1, -1), 9),
        ('TEXTCOLOR',   (0, 0), (0, -1), COLOR_DARK),
        ('TEXTCOLOR',   (1, 0), (1, -1), COLOR_MID_GRAY),
        ('TOPPADDING',  (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [COLOR_LIGHT_BG, colors.white]),
    ]))
    story.append(meta_table)

    story.append(PageBreak())

    # ─── PAGE 2 — TERRAIN ANALYSIS ───────────────────────────────────────────

    story.append(Paragraph('Terrain Analysis', S['section_heading']))
    story.append(HRFlowable(width=content_width, thickness=1, color=COLOR_BORDER))
    story.append(Spacer(1, 0.3 * cm))

    overlay_b64 = session_data.get('overlay_b64', '')
    if overlay_b64:
        try:
            img = _b64_to_rl_image(overlay_b64, content_width)
            story.append(img)
            story.append(Spacer(1, 0.3 * cm))
        except Exception:
            story.append(Paragraph('[Overlay image unavailable]', S['body']))

    # Traversability score
    score = traversability.get('score', 0)
    rating = traversability.get('rating', 'N/A')
    trav_rec = traversability.get('recommendation', '')

    score_color = COLOR_GREEN if score >= 70 else (COLOR_AMBER if score >= 40 else COLOR_RED)
    score_style = ParagraphStyle('score_dyn', parent=S['score_big'], textColor=score_color)
    rating_style = ParagraphStyle('rating_dyn', parent=S['rating'], textColor=score_color)

    score_data = [[
        Paragraph(f'{score:.1f}', score_style),
        Paragraph(f'{rating}\n{trav_rec}', rating_style),
    ]]
    score_table = Table(score_data, colWidths=[content_width * 0.25, content_width * 0.75])
    score_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.4 * cm))

    # Recommendation box
    action = recommendation.get('primary_action', 'N/A')
    reasoning = recommendation.get('reasoning', '')
    action_fg, action_bg = _action_color(action)

    rec_data = [[
        Paragraph(f'PRIMARY ACTION: {action}', ParagraphStyle(
            'rec_action', parent=S['body'],
            fontName='Helvetica-Bold', fontSize=11, textColor=action_fg,
        )),
    ], [
        Paragraph(reasoning, S['body']),
    ]]
    rec_table = Table(rec_data, colWidths=[content_width])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), action_bg),
        ('BACKGROUND',  (0, 1), (-1, 1), colors.white),
        ('BOX',         (0, 0), (-1, -1), 1.5, action_fg),
        ('TOPPADDING',  (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(rec_table)

    story.append(PageBreak())

    # ─── PAGE 3 — TERRAIN COMPOSITION ────────────────────────────────────────

    story.append(Paragraph('Terrain Composition', S['section_heading']))
    story.append(HRFlowable(width=content_width, thickness=1, color=COLOR_BORDER))
    story.append(Spacer(1, 0.3 * cm))

    comp_header = ['Class', 'Coverage %', 'Traversable', 'Status']
    comp_rows = [comp_header]
    alert_row_indices = []

    visible = [
        (cls_id, cls_data) for cls_id, cls_data in sorted(per_class.items())
        if cls_id not in ABSENT_CLASS_IDS
        and cls_data.get('percentage', 0) > 0
        and cls_data.get('present_in_dataset', True)
    ]

    for i, (cls_id, cls_data) in enumerate(visible, start=1):
        name = cls_data.get('name', CLASS_NAMES.get(cls_id, str(cls_id)))
        pct  = cls_data.get('percentage', 0)
        trav = 'Yes' if cls_id in TRAVERSABLE_IDS else 'No'
        status = 'ALERT' if cls_data.get('alert') and pct > 0.5 else 'Active' if pct > 0 else '—'
        comp_rows.append([name, f'{pct:.2f}%', trav, status])
        if cls_data.get('alert') and pct > 0.5:
            alert_row_indices.append(i)

    comp_col_w = [content_width * r for r in [0.35, 0.22, 0.22, 0.21]]
    comp_table = Table(comp_rows, colWidths=comp_col_w)
    comp_style = [
        ('BACKGROUND',   (0, 0), (-1, 0), COLOR_DARK),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLOR_LIGHT_BG, colors.white]),
        ('GRID',         (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',  (0, 0), (-1, -1), 8),
    ]
    for row_i in alert_row_indices:
        comp_style.append(('BACKGROUND', (0, row_i), (-1, row_i), COLOR_AMBER_BG))
        comp_style.append(('TEXTCOLOR',  (3, row_i), (3, row_i), COLOR_RED))
        comp_style.append(('FONTNAME',   (3, row_i), (3, row_i), 'Helvetica-Bold'))
    comp_table.setStyle(TableStyle(comp_style))
    story.append(comp_table)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        '* Trees (class 5) and Water (class 6) are excluded — zero pixels across the entire '
        'FalconCloud synthetic training dataset. The model was not trained on these classes '
        'and cannot predict them.',
        S['footnote'],
    ))

    story.append(PageBreak())

    # ─── PAGE 4 — THREAT ASSESSMENT ──────────────────────────────────────────

    story.append(Paragraph('Threat Assessment', S['section_heading']))
    story.append(HRFlowable(width=content_width, thickness=1, color=COLOR_BORDER))
    story.append(Spacer(1, 0.3 * cm))

    threat_level = threat.get('threat_level', 'N/A')
    threat_color = _threat_color(threat_level)
    threat_badge_data = [[Paragraph(f'THREAT LEVEL: {threat_level}', ParagraphStyle(
        'tbadge', parent=S['body'],
        fontName='Helvetica-Bold', fontSize=14, textColor=colors.white,
        alignment=TA_CENTER,
    ))]]
    threat_badge = Table(threat_badge_data, colWidths=[content_width])
    threat_badge.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), threat_color),
        ('TOPPADDING',    (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(threat_badge)
    story.append(Spacer(1, 0.4 * cm))

    # Active alerts
    active_alerts = threat.get('alerts', [])
    if active_alerts:
        story.append(Paragraph('Active Alerts', S['section_heading']))
        for alert in active_alerts:
            alert_text = (
                f"{alert.get('name', '?')}  —  {alert.get('percentage', 0):.2f}% coverage\n"
                f"{alert.get('message', '')}"
            )
            alert_data = [[Paragraph(alert_text, S['alert_text'])]]
            alert_table = Table(alert_data, colWidths=[content_width])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, -1), COLOR_RED),
                ('TOPPADDING',    (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING',   (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(alert_table)
            story.append(Spacer(1, 0.15 * cm))
    else:
        story.append(Paragraph('No active alerts.', S['body']))

    story.append(Spacer(1, 0.3 * cm))

    # Zone lists
    blocked = recommendation.get('avoid_zones', [])
    safe    = recommendation.get('safe_zones', [])

    def zone_list(title, zones, fg):
        if not zones:
            return
        story.append(Paragraph(title, S['section_heading']))
        zone_strs = [f"Row {z['row']} Col {z['col']}  (score: {z['score']:.1f})" for z in zones]
        zone_data = [[Paragraph(s, S['body'])] for s in zone_strs]
        zt = Table(zone_data, colWidths=[content_width])
        zt.setStyle(TableStyle([
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ('TOPPADDING',    (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [COLOR_LIGHT_BG, colors.white]),
            ('BOX', (0, 0), (-1, -1), 1, fg),
        ]))
        story.append(zt)
        story.append(Spacer(1, 0.2 * cm))

    zone_list('Blocked / Avoid Zones', blocked, COLOR_RED)
    zone_list('Safe Zones', safe, COLOR_GREEN)

    story.append(PageBreak())

    # ─── PAGE 5 — TECHNICAL DETAILS ──────────────────────────────────────────

    story.append(Paragraph('Technical Details', S['section_heading']))
    story.append(HRFlowable(width=content_width, thickness=1, color=COLOR_BORDER))
    story.append(Spacer(1, 0.3 * cm))

    # Architecture table
    arch_data = [
        ['Parameter', 'Value'],
        ['Architecture',    'SegFormer-B2'],
        ['Encoder',         'Mix Transformer (MiT-B2)'],
        ['Decoder',         'All-MLP Head'],
        ['Input Resolution','512 × 512'],
        ['Output Classes',  '10 (8 present in dataset)'],
        ['Parameters',      '~25M'],
        ['Training Loss',   'CrossEntropy + Dice (weighted)'],
        ['Optimizer',       'AdamW + warmup scheduler'],
        ['Hard Mining',     'Qdrant vector store — 638 hard examples'],
    ]
    arch_table = Table(arch_data, colWidths=[content_width * 0.4, content_width * 0.6])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), COLOR_DARK),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',     (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLOR_LIGHT_BG, colors.white]),
        ('GRID',         (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',  (0, 0), (-1, -1), 8),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 0.5 * cm))

    # Per-class IoU table
    story.append(Paragraph('Per-Class IoU — Best Checkpoint (run_best.pth)', S['section_heading']))
    iou_header = ['Class ID', 'Class Name', 'IoU', 'Notes']
    iou_rows = [iou_header]
    absent_row_idxs = []
    for cls_id in range(10):
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        iou_val = CLASS_IOUs.get(cls_id)
        if cls_id in ABSENT_CLASS_IDS:
            iou_rows.append([str(cls_id), name, 'N/A', 'Not in training dataset'])
            absent_row_idxs.append(len(iou_rows) - 1)
        else:
            note = 'Rare class ★' if cls_id in (8, 9) else ''
            iou_rows.append([str(cls_id), name, f'{iou_val:.4f}' if iou_val else '—', note])

    iou_col_w = [content_width * r for r in [0.15, 0.30, 0.20, 0.35]]
    iou_table = Table(iou_rows, colWidths=iou_col_w)
    iou_style = [
        ('BACKGROUND',   (0, 0), (-1, 0), COLOR_DARK),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLOR_LIGHT_BG, colors.white]),
        ('GRID',         (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',  (0, 0), (-1, -1), 8),
    ]
    for row_i in absent_row_idxs:
        iou_style.append(('TEXTCOLOR', (0, row_i), (-1, row_i), COLOR_MID_GRAY))
        iou_style.append(('FONTNAME',  (2, row_i), (2, row_i), 'Helvetica-Oblique'))
    iou_table.setStyle(TableStyle(iou_style))
    story.append(iou_table)
    story.append(Spacer(1, 0.4 * cm))

    # Training summary
    story.append(Paragraph('Training Summary', S['section_heading']))
    train_data = [
        ['Metric', 'Value'],
        ['Total Epochs',         '22'],
        ['Train Images',         '2857'],
        ['Val Images',           '317'],
        ['Hard Examples Mined',  '638 (Qdrant)'],
        ['Best mIoU',            '0.6109'],
        ['Best Flowers IoU',     '0.9801'],
        ['Best Logs IoU',        '0.6504'],
        ['Dataset Source',       'FalconCloud synthetic — 8 of 10 classes present'],
    ]
    train_table = Table(train_data, colWidths=[content_width * 0.4, content_width * 0.6])
    train_table.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), COLOR_DARK),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',     (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLOR_LIGHT_BG, colors.white]),
        ('GRID',         (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',  (0, 0), (-1, -1), 8),
    ]))
    story.append(train_table)

    doc.build(story)
    return output_path
