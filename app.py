import streamlit as st
import librosa
import numpy as np
import tempfile
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Регистрация шрифта
FONT = 'Helvetica'
try:
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    FONT = 'DejaVu'
except:
    pass

st.set_page_config(page_title="ПЕСНИ ПАВЛОВЫХ", layout="wide", page_icon="🎵")
st.title("🎵 ПЕСНИ ПАВЛОВЫХ — Редактор аккордов")

# ---------- ШАГ 1: Загрузка ----------
st.header("1. Загрузи MP3 и вставь текст")

col1, col2 = st.columns(2)

with col1:
    uploaded_audio = st.file_uploader("Аудиофайл (MP3)", type=['mp3'])
    song_title = st.text_input("Название песни", "")
    song_key = st.text_input("Тональность", "C (До мажор)")
    song_bpm = st.text_input("Темп (BPM)", "")

with col2:
    default_text = """Вступление:
E B | C#m G#m | A E | F#m B

Куплет 1:
Кто-то парус поднимет выше,
Кто-то держит в руках штурвал,
Каждый здесь Божий зов услышал,
Он нас Сам в этот путь призвал.

Припев:
Не теряя цели,
Обходя шторма и мели,
Сохраняя веру,
Держи курс на небо!

Бридж:
Не смотри в глубину, там холод,
Там вершины подводных скал.

Припев(2 раза):..."""

    raw_text = st.text_area("Текст песни (с разделами):", value=default_text, height=350,
                            help="Пиши разделы с новой строки: Куплет 1:, Припев:, Бридж:, Проигрыш:, Кода:")

# ---------- ШАГ 2: Анализ аккордов ----------
st.header("2. Автоматический анализ аккордов")

if st.button("🎼 Проанализировать аккорды из аудио", type="primary", disabled=not uploaded_audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name

    with st.spinner("Анализирую гармонию..."):
        y, sr = librosa.load(audio_path, sr=22050, duration=240)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        templates = {
            '': [1,0,0,0,1,0,0,1,0,0,0,0], 'm': [1,0,0,1,0,0,0,1,0,0,0,0],
            '7': [1,0,0,0,1,0,0,1,0,0,1,0], 'm7': [1,0,0,1,0,0,0,1,0,0,1,0],
            'sus4': [1,0,0,0,0,1,0,1,0,0,0,0], 'sus2': [1,0,1,0,0,0,0,1,0,0,0,0],
            '6': [1,0,0,0,1,0,0,1,0,1,0,0], 'maj7': [1,0,0,0,1,0,0,1,0,0,0,1],
        }
        notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

        chords_seq = []
        prev = ""
        for i in range(0, chroma.shape[1], 20):
            fc = chroma[:, i:i+20].mean(axis=1)
            fc = fc / (np.linalg.norm(fc) + 1e-10)
            best, mx = "", 0.62
            for suf, tpl in templates.items():
                tpl = np.array(tpl) / (np.linalg.norm(tpl) + 1e-10)
                for s in range(12):
                    c = np.dot(fc, np.roll(tpl, s))
                    if c > mx:
                        mx = c
                        best = notes[s] + suf
            if best and best != prev:
                chords_seq.append(best)
                prev = best
        os.unlink(audio_path)

    st.session_state['detected_chords'] = chords_seq
    st.success(f"Найдено {len(chords_seq)} аккордов: {' → '.join(chords_seq)}")

# ---------- ШАГ 3: Редактор ----------
st.header("3. Расставь аккорды (редактор)")

# Парсим текст на секции
lines = [l.strip() for l in raw_text.split('\n')]
sections = []
cur = {"title": "Текст", "lines": []}

keywords = ['вступление', 'куплет', 'припев', 'бридж', 'проигрыш', 'кода', 'финал', 'интро']

for line in lines:
    low = line.lower().replace(':', '').strip()
    is_section = any(low.startswith(k) for k in keywords)
    if is_section:
        if cur["lines"] or cur["title"] != "Текст":
            sections.append(cur)
        cur = {"title": line.replace(':', '').strip(), "lines": []}
    else:
        if line:
            cur["lines"].append(line)
if cur["lines"] or cur["title"] != "Текст":
    sections.append(cur)

# Если есть распознанные аккорды — распределяем их как подсказку
detected = st.session_state.get('detected_chords', [])
chord_pool = list(detected)

st.info("💡 Для каждой строки впиши аккорды через пробел (например: E B C#m). Оставь пустым если нет аккордов.")

edited_sections = []
for si, sec in enumerate(sections):
    st.markdown(f"#### 🎵 {sec['title']}")
    
    edited_lines = []
    for li, line in enumerate(sec["lines"]):
        c1, c2 = st.columns([2, 3])
        with c1:
            # Автоподсказка аккорда
            hint = chord_pool.pop(0) if chord_pool else ""
            chord_val = st.text_input(
                "Аккорды",
                value=hint,
                key=f"ch_{si}_{li}",
                placeholder="E B C#m"
            )
        with c2:
            text_val = st.text_input(
                "Текст строки",
                value=line,
                key=f"tx_{si}_{li}"
            )
        edited_lines.append({"text": text_val, "chords": chord_val.strip()})
    
    edited_sections.append({"title": sec["title"], "lines": edited_lines})
    st.markdown("---")

# ---------- ШАГ 4: Генерация PDF ----------
st.header("4. Скачай PDF")

if st.button("📥 Сгенерировать PDF", type="primary"):
    buf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    c = canvas.Canvas(buf.name, pagesize=A4)
    w, h = A4
    y = h - 2*cm

    # Заголовок
    if song_title:
        c.setFont(FONT, 18)
        c.setFillColor(colors.black)
        c.drawCentredString(w/2, y, song_title.upper())
        y -= 0.8*cm

    # Тональность и темп
    info_parts = []
    if song_key:
        info_parts.append(f"Тональность: {song_key}")
    if song_bpm:
        info_parts.append(f"Темп: {song_bpm} BPM")
    if info_parts:
        c.setFont(FONT, 9)
        c.setFillColor(colors.grey)
        c.drawCentredString(w/2, y, " | ".join(info_parts))
        y -= 1*cm

    for sec in edited_sections:
        if y < 4*cm:
            c.showPage()
            y = h - 2*cm

        # Заголовок секции
        c.setFont(FONT, 11)
        c.setFillColor(colors.HexColor('#333333'))
        c.drawString(1*cm, y, sec["title"] + ":")
        y -= 0.6*cm

        for item in sec["lines"]:
            if y < 3*cm:
                c.showPage()
                y = h - 2*cm

            text = item["text"]
            chords_str = item["chords"]

            # Если многоточие — просто текст
            if text.startswith("...") or text.endswith("..."):
                c.setFont(FONT, 10)
                c.setFillColor(colors.grey)
                c.drawString(1*cm, y, text)
                y -= 0.6*cm
                continue

            # Аккорды над строкой
            if chords_str:
                c.setFont(FONT, 10)
                c.setFillColor(colors.HexColor('#00008B'))
                c.drawString(1.2*cm, y + 13, chords_str)

            # Текст строки
            c.setFont(FONT, 11)
            c.setFillColor(colors.black)
            c.drawString(1.2*cm, y, text)
            y -= 0.75*cm

        y -= 0.4*cm

    # Подпись
    c.setFillColor(colors.grey)
    c.setFont(FONT, 8)
    c.drawCentredString(w/2, 1*cm, 'ПЕСНИ ПАВЛОВЫХ\u00A9')
    c.save()

    with open(buf.name, 'rb') as f:
        pdf_bytes = f.read()
    os.unlink(buf.name)

    st.success("✅ PDF готов!")
    st.download_button(
        label="📥 Скачать PDF",
        data=pdf_bytes,
        file_name=f"{song_title or 'song'}_pavlovy.pdf",
        mime="application/pdf"
    )
