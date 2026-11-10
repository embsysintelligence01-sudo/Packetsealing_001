"""Fixed-page visual session audit matching Jewel Tracker's navy/teal report."""
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Paragraph

NAVY = colors.HexColor('#0c1c2b')
TEAL = colors.HexColor('#008b8b')
PALE = colors.HexColor('#f3f6f9')
BORDER = colors.HexColor('#d6e1eb')
INK = colors.HexColor('#25384b')


def write_session_report(path, directory, work, flow, result, finish_t, clock_text):
    """One overview and one page per jewel (three pages for two jewels)."""
    width, height = A4
    margin, usable = 28, width-56
    canvas = pdfcanvas.Canvas(str(path), pagesize=A4, pageCompression=1)
    canvas.setTitle('Packet sealing | Session audit')
    canvas.setAuthor('EMBSYS / PACKET SEALING')
    number = result.get('packet_number', {})
    packet = number.get('text') if number.get('status') == 'confirmed' else 'Not confirmed'
    records = result['jewels']
    starts = [j.started for j in flow.jewels.values() if j.started is not None]
    start = min(starts) if starts else None
    page = 0

    def text(value, x, top, size=9, color=INK, bold=False, limit=usable):
        style = ParagraphStyle('audit', fontName='Helvetica-Bold' if bold else 'Helvetica',
                               fontSize=size, leading=size*1.35, textColor=color)
        p = Paragraph(escape(str(value).replace('—', '-')).replace('\n', '<br/>'), style)
        _, h = p.wrap(limit, 1000)
        p.drawOn(canvas, x, top-h)
        return h

    def card(x, top, w, h):
        canvas.setFillColor(PALE)
        canvas.setStrokeColor(BORDER)
        canvas.roundRect(x, top-h, w, h, 8, fill=1, stroke=1)

    def picture(source, x, top, w, h):
        source = Path(source)
        if not source.is_file():
            text('Evidence not captured', x+6, top-12, color=TEAL, limit=w-12)
            return
        image = ImageReader(str(source))
        iw, ih = image.getSize()
        scale = min(w/iw, h/ih)
        dw, dh = iw*scale, ih*scale
        canvas.drawImage(image, x+(w-dw)/2, top-h+(h-dh)/2, dw, dh, mask='auto')

    def button(name, label, x, top, w=135):
        canvas.setFillColor(colors.white)
        canvas.setStrokeColor(BORDER)
        canvas.roundRect(x, top-27, w, 27, 8, fill=1, stroke=1)
        text('Open '+label if name else label+': unavailable', x+8, top-7, 8, TEAL, limit=w-16)
        if name:
            canvas.linkURL(name, (x, top-27, x+w, top), relative=0, thickness=0)

    def timestamp(t):
        if t is None:
            return 'Not observed'
        return datetime.fromtimestamp(t, ZoneInfo('Asia/Kolkata')).strftime('%H:%M:%S.%f')[:-3]+' IST'

    def header(title, subtitle):
        nonlocal page
        page += 1
        canvas.setFillColor(NAVY)
        canvas.rect(0, height-108, width, 108, fill=1, stroke=0)
        text('EMBSYS / PACKET SEALING', margin, height-18, 9, colors.white, True)
        text(title, margin, height-40, 19, colors.white, True)
        text(subtitle, margin, height-70, 8, colors.white)
        text('Packet number: '+str(packet), margin, height-88, 9, colors.white, True)
        canvas.setFillColor(TEAL)
        canvas.rect(0, height-111, width, 3, fill=1, stroke=0)
        text('Session '+directory.name, margin, 25, 7)
        canvas.setFillColor(INK)
        canvas.setFont('Helvetica', 7)
        canvas.drawRightString(width-margin, 16, f'VISUAL AUDIT | {page:02d}')

    def sealing_label():
        return 'SEALED (INFERRED)' if flow.sealed else 'INCOMPLETE / REVIEW'

    # Up to three cards fit on each overview. Two-jewel sessions have one overview.
    chunks = [records[i:i+3] for i in range(0, len(records), 3)] or [[]]
    for batch_index, batch in enumerate(chunks):
        header('Packet sealing | Session audit',
               'All jewels | Start: '+clock_text(start)+' | End: '+timestamp(finish_t))
        top = height-130
        stat_width = (usable-16)/3
        placed = sum(j.status == 'Placed in packet — inferred' for j in flow.jewels.values())
        for i, (label, value) in enumerate([('JEWELS', len(flow.jewels)),
                                          ('PLACEMENTS INFERRED', placed), ('SEALING', sealing_label())]):
            x = margin+i*(stat_width+8)
            card(x, top, stat_width, 54)
            text(label, x+10, top-9, 8)
            text(value, x+10, top-28, 10 if i == 2 else 18, TEAL, True, stat_width-20)
        top -= 69
        text('JEWEL PLACEMENT OUTCOMES', margin, top, 10, bold=True)
        top -= 24
        for record in batch:
            jid = record['id']
            j = flow.jewels[jid]
            card(margin, top, usable, 92)
            text(f'J{jid}', margin+11, top-12, 12, TEAL, True)
            picture(work / f'J{jid}_initial.jpg', margin+45, top-8, 56, 56)
            text(j.status, margin+115, top-12, 10, bold=True, limit=usable-275)
            text('Recording review: '+('; '.join(record['evidence_issues']) or 'No coverage issue found'),
                 margin+115, top-36, 8, limit=usable-275)
            button(record.get('video'), f'J{jid} transfer video', width-margin-143, top-16, 133)
            text('Taken: '+timestamp(j.started)+' | Placement: '+timestamp(j.ended),
                 margin+11, top-72, 8, limit=usable-22)
            top -= 101
        top -= 4
        card(margin, top, usable, 87)
        text('PACKET NUMBER / OCR', margin+11, top-9, 9, TEAL, True)
        text('Packet number: '+str(packet), margin+11, top-28, 12, bold=True, limit=usable-180)
        confidence = number.get('confidence')
        score = f'{confidence:.1%}' if confidence is not None else 'Not confirmed'
        text('Captured: '+timestamp(number.get('confirmed_at'))+' | Confidence: '+score,
             margin+11, top-53, 8, limit=usable-180)
        if number.get('status') == 'confirmed':
            picture(work / 'packet_number_crop.jpg', width-margin-158, top-21, 145, 45)
        top -= 101
        text('Sealing: '+sealing_label()+' | Strip removed: '+timestamp(flow.strip_removed_at),
             margin, top, 9, bold=True)
        button(result.get('sealing_video'), 'packet sealing video', margin, top-25, 160)
        text('Last updated: '+result['report_updated'], margin+173, top-31, 8, limit=usable-173)
        text('Status: '+result['status'], margin, top-64, 8)
        text('Placement follows the pickup-owned transfer. Sealing is inferred from verified strip removal; '
             'adhesive closure and strength are not measured. Packet rotation itself is not verified.',
             margin, 87, 8)
        canvas.showPage()

    for record in records:
        j = flow.jewels[record['id']]
        clip_end = j.clip_ended if j.clip_ended is not None else finish_t
        header(f'J{j.id} | Packing evidence',
               'Taken: '+timestamp(j.started)+' | Placement: '+timestamp(j.ended))
        top = height-127
        card(margin, top, usable, 86)
        picture(work / f'J{j.id}_initial.jpg', margin+10, top-8, 70, 70)
        text('ASSIGNED JEWEL / CROPPED PHOTO', margin+96, top-10, 9, TEAL, True)
        text(f'J{j.id} | '+j.status, margin+96, top-30, 10, bold=True, limit=usable-265)
        text('Pickup / recording started: '+timestamp(j.started), margin+96, top-54, 8, limit=usable-110)
        button(record.get('video'), f'J{j.id} transfer video', width-margin-149, top-14, 139)
        top -= 104
        gap = 14
        panel_w = (usable-gap)/2
        for index, (label, suffix, event_t) in enumerate([
                ('01 / Pickup', 'pickup', j.started),
                ('02 / Packet interaction', 'entry', j.entry)]):
            x = margin+index*(panel_w+gap)
            text(label, x, top, 10, TEAL, True, panel_w)
            picture(work / f'J{j.id}_{suffix}.jpg', x, top-24, panel_w, 140)
            text(timestamp(event_t), x, top-171, 8, limit=panel_w)
        top -= 202
        text('03 / Placement result retained', margin, top, 10, TEAL, True)
        picture(work / f'J{j.id}_after.jpg', margin, top-24, usable, 173)
        text(timestamp(j.ended)+' | Placement inferred from this transfer.', margin, top-204, 8)
        top -= 231
        card(margin, top, usable, 79)
        text('RECORDING / PLACEMENT', margin+10, top-8, 9, TEAL, True)
        text('Recording handoff / finish: '+timestamp(clip_end), margin+10, top-26, 8)
        text('Recording review: '+('; '.join(record['evidence_issues']) or 'No coverage issue found'),
             margin+10, top-42, 8, limit=usable-20)
        text(j.note, margin+10, top-57, 7, limit=usable-20)
        canvas.showPage()
    sealing_start = result.get('sealing_started_at')
    sealing_end = result.get('sealing_ended_at')
    header('Sealed packet | Sealing evidence',
           'Sealing started: '+timestamp(sealing_start)+' | Ended: '+timestamp(sealing_end))
    top = height-130
    card(margin, top, usable, 96)
    text(sealing_label(), margin+12, top-12, 12, TEAL, True)
    text('Sealing started: '+clock_text(sealing_start), margin+12, top-37, 10)
    text('Sealing ended: '+clock_text(sealing_end), margin+12, top-57, 10)
    if sealing_start is not None and sealing_end is not None:
        text(f'Sealing duration: {max(0., sealing_end-sealing_start):.1f} seconds',
             margin+12, top-77, 9)
    top -= 117
    text('SEALED PACKET IMAGE', margin, top, 11, TEAL, True)
    # Use the actual sealing-time frame, never the later rotated/OCR frame.
    picture(work / 'sealed.jpg', margin, top-27, usable, 320)
    text('Sealing image captured: '+timestamp(sealing_end), margin, top-362, 9)
    text('Sealing starts when all jewels have been placed. Sealing ends when strip removal '
         'is confirmed; the later rotation and OCR stage is recorded separately.',
         margin, top-387, 9)
    button(result.get('sealing_video'), 'packet sealing video', margin, top-431, 170)
    text('Sealing is inferred from verified strip removal; adhesive closure and strength are not measured.',
         margin, 80, 8)
    canvas.showPage()
    canvas.save()
