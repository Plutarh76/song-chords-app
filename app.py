import streamlit as st
import tempfile
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Регистрация шрифта с кириллицей
FONT = 'Helvetica'
try:
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    FONT = 'DejaVu'
except:
    pass

st.set_page_config(page_title="ПЕСНИ ПАВЛОВЫХ", layout="wide", page_icon="🎵")
st.title("🎵 ПЕСНИ ПАВЛОВЫХ — Оформление PDF")
st.markdown("Вставь текст и аккорды → получи PDF как в образцах")

# ---------- ВВОД ДАННЫХ ----------
st.header("1. Данные песни")

col1, col2 = st.columns(2)

with col1:
    song_title = st.text_input("Название песни", "Курс на небо")
    key = st.text_input("Тональность", "E (Ми мажор)")
    tempo = st.text_input("Темп", "120 BPM")

with col2:
    intro_chords = st.text_input("Вступление (аккорды через |)", "E B | C#m G#m | A E | F#m B")
    outro_chords = st.text_input("Проигрыш/Кода (необязательно)", "")

# Текст песни
default_text = """Куплет 1:
Кто-то парус поднимет выше,
Кто-то держит в руках штурвал,
Каждый здесь Божий зов услышал,
Он нас Сам в этот путь призвал.

Куплет 2:
Курс на вечность – одна задача,
Мир меняется в играх волн.
У штурвала стоять — не значит
Закрывать горизонт собой.

Припев:
Не теряя цели,
Обходя шторма и мели,
Сохраняя веру,
Держи курс на небо!

Припев(2 раза):...

Куплет 3:
Не смотри в глубину, там холод,
Там вершины подводных скал.
Там обломки тех, кто расколот,
Кто не в небе свой путь искал.

Припев(2 раза):...

Проигрыш:
E B | C#m G#m | A E | F#m B

Куплет 4:
Пусть не видно в тумане море,
Но глаза свои подними
На Звезду, что не гаснет в шторме,
На Любовь, что не знает тьмы.

Припев(2 раза):...

Концовка:
E B | E"""

raw_text = st.text_area("Текст песни:", value=default_text, height=400,
                        help="Формат: Раздел:\\nстрока1\\nстрока2\\n\\nРаздел2:\\n...")

# Аккорды для каждой строки (через запятую для каждой строки)
st.header("2. Аккорды для каждой строки")
st.info(" Впиши аккорды для каждой строки через запятую. Оставь пустым если аккордов нет.")

# Парсим текст на секции и строки
lines_list = [l.strip() for l in raw_text.split('\n') if l.strip()]
sections = []
cur = {"title": "", "lines": []}
keywords = ['вступление', 'куплет', 'припев', 'бридж', 'проигрыш', 'кода', 'финал', 'интро', 'концовка']

for line in lines_list:
    low = line.lower().replace(':', '').strip()
    if any(low.startswith(k) for k in keywords) and line.endswith(':'):
        if cur["lines"]:
            sections.append(cur)
        cur = {"title": line, "lines": []}
    elif line:
        cur["lines"].append(line)
if cur["lines"]:
    sections.append(cur)

# Поля для аккордов
chords_input = {}
for si, section in enumerate(sections):
    with st.expander(f"🎵 {section['title']}", expanded=True):
        for li, line_text in enumerate(section["lines"]):
            key_id = f"{si}_{li}"
            chords_input[key_id] = st.text_input(
                f"{line_text[:50]}...",
                value="",
                key=key_id,
                placeholder="E B C#m"
            )

# ---------- ГЕНЕРАЦИЯ PDF ----------
st.header("3. Генерация PDF")

if st.button("👁️ Предпросмотр и скачивание", type="primary"):
    buf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    c = canvas.Canvas(buf.name, pagesize=A4)
    w, h = A4
    y = h - 2*cm
    
    # Заголовок песни (крупно, капсом)
    c.setFont(FONT, 18)
    c.setFillColor(colors.black)
    c.drawCentredString(w/2, y, song_title.upper())
    y -= 1*cm
    
    # Тональность и темп
    info = []
    if key:
        info.append(f"Тональность: {key}")
    if tempo:
        info.append(f"Темп: {tempo}")
    if info:
        c.setFont(FONT, 10)
        c.setFillColor(colors.grey)
        c.drawCentredString(w/2, y, " | ".join(info))
        y -= 1*cm
    
    # Вступление
    if intro_chords:
        if y < 3*cm:
            c.showPage()
            y = h - 2*cm
        c.setFont(FONT, 11)
        c.setFillColor(colors.black)
        c.drawString(1*cm, y, f"Вступление: {intro_chords}")
        y -= 0.8*cm
    
    # Секции
    for si, section in enumerate(sections):
        if y < 3*cm:
            c.showPage()
            y = h - 2*cm
        
        # Заголовок секции
        c.setFont(FONT, 12)
        c.setFillColor(colors.black)
        c.drawString(1*cm, y, section["title"])
        y -= 0.7*cm
        
        # Строки
        for li, line_text in enumerate(section["lines"]):
            if y < 2.5*cm:
                c.showPage()
                y = h - 2*cm
            
            key_id = f"{si}_{li}"
            chords_str = chords_input.get(key_id, "")
            
            # Если многоточие — просто текст серым
            if "..." in line_text:
                c.setFont(FONT, 11)
                c.setFillColor(colors.grey)
                c.drawString(1*cm, y, line_text)
                y -= 0.7*cm
                continue
            
            # Аккорды над строкой
            if chords_str:
                c.setFont(FONT, 11)
                c.setFillColor(colors.HexColor('#00008B'))
                # Рисуем аккорды с равномерным распределением
                chords_list = [ch.strip() for ch in chords_str.split() if ch.strip()]
                if chords_list:
                    # Вычисляем ширину строки текста
                    text_width = c.stringWidth(line_text, FONT, 12)
                    start_x = 1*cm
                    # Распределяем аккорды равномерно
                    if len(chords_list) == 1:
                        c.drawString(start_x, y + 14, chords_list[0])
                    else:
                        spacing = text_width / (len(chords_list) - 1) if len(chords_list) > 1 else 0
                        for ci, chord in enumerate(chords_list):
                            x_pos = start_x + (ci * spacing) if len(chords_list) > 1 else start_x
                            c.drawString(x_pos, y + 14, chord)
            
            # Текст строки
            c.setFillColor(colors.black)
            c.setFont(FONT, 12)
            c.drawString(1*cm, y, line_text)
            y -= 0.8*cm
        
        y -= 0.4*cm
    
    # Проигрыш/Кода
    if outro_chords:
        if y < 3*cm:
            c.showPage()
            y = h - 2*cm
        c.setFont(FONT, 11)
        c.setFillColor(colors.black)
        c.drawString(1*cm, y, f"Проигрыш/Кода: {outro_chords}")
        y -= 0.8*cm
    
    # Подпись
    c.setFillColor(colors.grey)
    c.setFont(FONT, 9)
    c.drawCentredString(w/2, 1*cm, 'ПЕСНИ ПАВЛОВЫХ©')
    c.save()
    
    with open(buf.name, 'rb') as f:
        pdf_bytes = f.read()
    
    # Превью
    import base64
    pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
    st.markdown("### 📄 Предварительный просмотр:")
    st.components.v1.html(
        f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="700px"></iframe>',
        height=700
    )
    
    st.download_button(
        label="📥 Скачать PDF",
        data=pdf_bytes,
        file_name=f"{song_title}_pavlovy.pdf",
        mime="application/pdf",
        type="primary"
    )
    
    os.unlink(buf.name)
