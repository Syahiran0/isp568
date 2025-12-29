import io
import time
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def generate_student_report_pdf(
    inputs: Dict[str, float],
    evaluation: Dict[str, Any],
    suggestion: str
) -> io.BytesIO:

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    # ================= COLORS =================
    DARK_GREEN = colors.HexColor("#1F4D3A")
    LIGHT_GREEN = colors.HexColor("#E3F4EA")
    TEXT_GREEN = colors.HexColor("#144C3A")
    GREY = colors.HexColor("#666666")

    # ================= STYLES =================
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='TitleCenter',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=10,
        fontName='Helvetica-Bold',
        textColor=TEXT_GREEN
    ))

    styles.add(ParagraphStyle(
        name='Heading',
        fontSize=11,
        leading=14,
        fontName='Helvetica-Bold',
        textColor=colors.white
    ))

    styles.add(ParagraphStyle(
        name='NormalLeft',
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        textColor=TEXT_GREEN
    ))

    styles.add(ParagraphStyle(
        name='SmallCenter',
        fontSize=9,
        alignment=TA_CENTER,
        textColor=GREY
    ))

    story: List[Any] = []

    # ================= HEADER =================
    header = Table([
        [
            Paragraph(
                "COLLEGE OF COMPUTING, INFORMATICS AND MATHEMATICS<br/><br/>"
                "BACHELOR OF INFORMATION SYSTEMS (HONS.) INTELLIGENT SYSTEMS ENGINEERING",
                styles['TitleCenter']
            ),
        ]
    ], colWidths=[430, 120])

    header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    story.append(header)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        f"Report Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        styles['SmallCenter']
    ))
    story.append(Spacer(1, 20))

    # ================= STUDENT INPUT SCORES =================
    input_table = Table([
        ["Student Input Scores", ""],
        ["Attendance", f"{inputs['attendance']:.1f}%"],
        ["Test Score", f"{inputs['test_score']:.1f}%"],
        ["Assignment Score", f"{inputs['assignment_score']:.1f}%"],
        ["Ethics", f"{inputs.get('ethics', 0.0):.1f}%"],
        ["Cognitive Skills", f"{inputs.get('cognitive', 0.0):.1f}%"],
    ], colWidths=[260, 260])

    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_GREEN),
        ('SPAN', (0, 0), (-1, 0)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GREEN),
        ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))

    story.append(input_table)
    story.append(Spacer(1, 20))

    # ================= EVALUATION RESULTS =================
    level = evaluation['performance_level']
    score = evaluation['fuzzy_score']

    eval_table = Table([
        ["Evaluation Results (Fuzzy Logic)", ""],
        ["Fuzzy Score", f"{score:.2f} / 100"],
        ["Performance Level", level.upper()],
    ], colWidths=[260, 260])

    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_GREEN),
        ('SPAN', (0, 0), (-1, 0)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GREEN),
        ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))

    story.append(eval_table)
    story.append(Spacer(1, 20))

    # ================= AI SUGGESTION =================
    suggestion_table = Table([
        ["Academic Advisor's Suggestion"],
        [Paragraph(suggestion.replace('\n', '<br/>'), styles['NormalLeft'])]
    ], colWidths=[520])

    suggestion_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_GREEN),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GREEN),
        ('BOX', (0, 0), (-1, -1), 1, colors.white),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 40),
    ]))

    story.append(suggestion_table)
    story.append(Spacer(1, 20))

    # ================= FOOTER =================
    story.append(Paragraph(
        "Note: This evaluation is based on fuzzy logic analysis and AI-generated academic suggestions.",
        styles['SmallCenter']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer
