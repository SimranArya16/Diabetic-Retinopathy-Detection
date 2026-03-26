import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime

DR_CLASSES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

RISK_LEVELS = {0: 'Low', 1: 'Low-Moderate', 2: 'Moderate',
               3: 'High', 4: 'Very High'}

DR_DESCRIPTIONS = {
    0: ("No signs of diabetic retinopathy detected. "
        "Regular annual screening is recommended."),
    1: ("Mild non-proliferative diabetic retinopathy. "
        "Microaneurysms are present. Follow-up in 9-12 months."),
    2: ("Moderate non-proliferative diabetic retinopathy. "
        "Multiple microaneurysms and hemorrhages present. "
        "Follow-up in 6 months."),
    3: ("Severe non-proliferative diabetic retinopathy. "
        "Significant hemorrhages in all quadrants. "
        "Urgent referral within 1 month."),
    4: ("Proliferative diabetic retinopathy. "
        "Neovascularization detected. Immediate referral required."),
}


def save_combined_figure(original_img, gradcam_overlay, lime_overlay,
                          class_idx, probs, active_regions, output_path):
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original_img)
    ax1.set_title('Original Fundus Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(gradcam_overlay)
    ax2.set_title('Grad-CAM++ Heatmap', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(lime_overlay)
    ax3.set_title('LIME Explanation', fontsize=12, fontweight='bold')
    ax3.axis('off')

    plt.suptitle(
        f'Prediction: {DR_CLASSES[class_idx]} '
        f'(Confidence: {probs[class_idx]*100:.1f}%)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


def generate_pdf_report(patient_id, image_name, figure_path,
                         class_idx, probs, active_regions,
                         output_path='report.pdf'):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                         Spacer, Image as RLImage,
                                         Table, TableStyle)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc    = SimpleDocTemplate(output_path, pagesize=letter,
                                   topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        story.append(Paragraph(
            "Diabetic Retinopathy Screening Report",
            styles['Title']))
        story.append(Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')} "
            f"| Patient ID: {patient_id}",
            styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # Summary table
        risk = RISK_LEVELS[class_idx]
        table_data = [
            ['Parameter',          'Value'],
            ['Predicted DR Stage', DR_CLASSES[class_idx]],
            ['Model Confidence',   f"{probs[class_idx]*100:.1f}%"],
            ['Risk Level',         risk],
            ['Regions of Concern', ', '.join(active_regions)
                                   if active_regions else 'N/A'],
        ]
        tbl = Table(table_data, colWidths=[2.5*inch, 4*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor('#ecf0f1'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.2*inch))

        # Clinical description
        story.append(Paragraph("Clinical Interpretation", styles['Heading2']))
        story.append(Paragraph(DR_DESCRIPTIONS[class_idx], styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # XAI figure
        story.append(Paragraph("AI Explainability Analysis", styles['Heading2']))
        story.append(Paragraph(
            "Grad-CAM++ heatmap highlights retinal regions influencing "
            "the prediction. Warmer colors indicate higher model attention.",
            styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(RLImage(figure_path, width=6.5*inch, height=2.2*inch))
        story.append(Spacer(1, 0.2*inch))

        # Probability table
        story.append(Paragraph("Class Probabilities", styles['Heading2']))
        prob_data = [['Stage', 'Probability']] + [
            [DR_CLASSES[i], f"{probs[i]*100:.2f}%"] for i in range(5)
        ]
        prob_tbl = Table(prob_data, colWidths=[3*inch, 3*inch])
        prob_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor('#ecf0f1'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, class_idx+1), (-1, class_idx+1),
             colors.HexColor('#fadbd8')),
            ('FONTNAME', (0, class_idx+1), (-1, class_idx+1),
             'Helvetica-Bold'),
        ]))
        story.append(prob_tbl)
        story.append(Spacer(1, 0.2*inch))

        # Disclaimer
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI system for "
            "screening assistance only. It is not a substitute for clinical "
            "diagnosis. All findings must be reviewed by a qualified "
            "ophthalmologist.",
            styles['Normal']))

        doc.build(story)

    except ImportError:
        # Fallback if reportlab not installed
        with open(output_path.replace('.pdf', '.txt'), 'w') as f:
            f.write(f"DR Report — Patient: {patient_id}\n")
            f.write(f"Prediction: {DR_CLASSES[class_idx]}\n")
            f.write(f"Confidence: {probs[class_idx]*100:.1f}%\n")
            f.write(f"Risk: {RISK_LEVELS[class_idx]}\n")
            f.write(f"Description: {DR_DESCRIPTIONS[class_idx]}\n")

    return output_path