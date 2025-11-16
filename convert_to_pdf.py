"""
Script to convert Theoretical Analysis markdown to PDF using reportlab
"""
import os
import sys
import re

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install reportlab -q")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

def parse_markdown(md_content):
    """Parse markdown content and return structured elements"""
    elements = []
    lines = md_content.split('\n')
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20,
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15,
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=8,
        spaceBefore=12,
    )
    
    h4_style = ParagraphStyle(
        'CustomH4',
        parent=styles['Heading4'],
        fontSize=12,
        textColor=colors.HexColor('#555555'),
        spaceAfter=6,
        spaceBefore=10,
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8,
        alignment=TA_JUSTIFY,
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        textColor=colors.HexColor('#c7254e'),
        backColor=colors.HexColor('#f4f4f4'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=8,
    )
    
    i = 0
    in_code_block = False
    code_lines = []
    
    while i < len(lines):
        line = lines[i]
        
        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End of code block
                if code_lines:
                    code_text = '\n'.join(code_lines)
                    elements.append(Preformatted(code_text, code_style))
                    code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue
        
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
        
        # Handle horizontal rules
        if line.strip() == '---' or line.strip().startswith('---'):
            elements.append(Spacer(1, 0.3*inch))
            i += 1
            continue
        
        # Handle headings
        if line.startswith('# '):
            text = line[2:].strip()
            elements.append(Paragraph(escape_html(text), title_style))
            elements.append(Spacer(1, 0.2*inch))
        elif line.startswith('## '):
            text = line[3:].strip()
            elements.append(Paragraph(escape_html(text), h1_style))
            elements.append(Spacer(1, 0.15*inch))
        elif line.startswith('### '):
            text = line[4:].strip()
            elements.append(Paragraph(escape_html(text), h2_style))
            elements.append(Spacer(1, 0.1*inch))
        elif line.startswith('#### '):
            text = line[5:].strip()
            elements.append(Paragraph(escape_html(text), h3_style))
            elements.append(Spacer(1, 0.08*inch))
        elif line.startswith('##### '):
            text = line[6:].strip()
            elements.append(Paragraph(escape_html(text), h4_style))
            elements.append(Spacer(1, 0.06*inch))
        # Handle lists
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            text = line.strip()[2:].strip()
            # Convert markdown bold/italic
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            text = re.sub(r'`(.*?)`', r'<font face="Courier" color="#c7254e">\1</font>', text)
            elements.append(Paragraph(f"• {escape_html(text)}", normal_style))
        # Handle numbered lists
        elif re.match(r'^\d+\.\s', line):
            text = re.sub(r'^\d+\.\s', '', line).strip()
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            elements.append(Paragraph(f"• {escape_html(text)}", normal_style))
        # Handle tables (simple detection)
        elif '|' in line and i > 0 and '|' in lines[i-1]:
            # Try to parse table
            table_data = []
            header_line = lines[i-1]
            if '|' in header_line:
                headers = [h.strip() for h in header_line.split('|') if h.strip()]
                if headers:
                    table_data.append(headers)
                    # Check next line for separator
                    if i+1 < len(lines) and '---' in lines[i+1]:
                        i += 1
                    # Get data rows
                    j = i + 1
                    while j < len(lines) and '|' in lines[j]:
                        row = [cell.strip() for cell in lines[j].split('|') if cell.strip()]
                        if row:
                            table_data.append(row)
                        j += 1
                    if len(table_data) > 1:
                        t = Table(table_data)
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                        ]))
                        elements.append(t)
                        elements.append(Spacer(1, 0.2*inch))
                        i = j
                        continue
        # Handle regular paragraphs
        elif line.strip():
            text = line.strip()
            # Convert markdown formatting
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            text = re.sub(r'`(.*?)`', r'<font face="Courier" color="#c7254e">\1</font>', text)
            elements.append(Paragraph(escape_html(text), normal_style))
        else:
            # Empty line
            elements.append(Spacer(1, 0.1*inch))
        
        i += 1
    
    return elements

def escape_html(text):
    """Escape HTML special characters"""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text

def markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF"""
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Parse markdown
    elements = parse_markdown(md_content)
    
    # Create PDF
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Build PDF
    try:
        doc.build(elements)
        print(f"Successfully created PDF: {pdf_file}")
        return True
    except Exception as e:
        print(f"Error creating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    md_file = "Theoretical_Analysis_AI_Technologies.md"
    pdf_file = "Theoretical_Analysis_AI_Technologies.pdf"
    
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found!")
        sys.exit(1)
    
    print(f"Converting {md_file} to PDF...")
    print("This may take a moment...")
    success = markdown_to_pdf(md_file, pdf_file)
    
    if success:
        print(f"\nPDF created successfully: {pdf_file}")
        print(f"  Location: {os.path.abspath(pdf_file)}")
    else:
        print("\nFailed to create PDF")
        sys.exit(1)
