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
from reportlab.lib.fonts import addMapping

# Регистрация шрифта
FONT = 'Helvetica'
try:
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    FONT = 'DejaVu'
except:
    pass

st.set_page_config(page_title="ПЕСНИ ПАВЛОВЫХ", layout="wide", page_icon="🎵")
st.title("🎵 ПЕСНИ ПАВЛОВЫХ — Редактор аккордов")

# Инициализация
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'sections_data' not in st.session_state:
    st.session_state.sections_data = []
if 'song_title' not in st.session_state:
    st.session_state.song_title = ""
if 'key_detected' not in st.session_state:
    st.session_state.key_detected = ""

# ШАГ 1: Загрузка
st.header("1. Загрузка")
col1, col2 = st.columns(2)

with col1:
    uploaded_audio = st.file_uploader("🎵 Аудио (MP3)", type=['mp3'])
    song_title = st.text_input("Название", "Без названия")

with col2:
    default_text = """Куплет 1:
Мир в смятении расколот
Царь въезжает в древний город
Благословен Грядущий Царь царей

Припев:
Древний город, город мира, Иерусалим
Сколько раз хотел укрыть Я вас крылом своим"""
    
    raw_text = st.text_area("📝 Текст:", value=default_text, height=250)

# ШАГ 2: Анализ
if st.button("🎼 Анализировать", type="primary", disabled=not uploaded_audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name
    
    with st.spinner("⏳ Анализирую..."):
        try:
            # Анализ аккордов
            y, sr = librosa.load(audio_path, sr=22050, duration=240)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            templates = {
                '': [1,0,0,0,1,0,0,1,0,0,0,0], 'm': [1,0,0,1,0,0,0,1,0,0,0,0],
                '7': [1,0,0,0,1,0,0,1,0,0,1,0], 'm7': [1,0,0,1,0,0,0,1,0,0,1,0],
                'sus4': [1,0,0,0,0,1,0,1,0,0,0,0], 'sus2': [1,0,1,0,0,0,0,1,0,0,0,0],
                'maj7': [1,0,0,0,1,0,0,1,0,0,0,1],
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
            
            chord_counts = {}
            for c in chords_seq:
                chord_counts[c] = chord_counts.get(c, 0) + 1
            major = [c for c in chord_counts if not c.endswith('m') and not c.endswith('7')]
            key = f"{max(major, key=lambda x: chord_counts[x])} (мажор)" if major else f"{max(chord_counts, key=chord_counts.get)} (минор)"
            
            st.session_state.key_detected = key
            
            # Парсим текст
            lines = [l.strip() for l in raw_text.split('\n')]
            sections = []
            cur = {"title": "Текст", "lines": []}
            keywords = ['вступление', 'куплет', 'припев', 'бридж', 'проигрыш', 'кода', 'финал', 'интро']
            
            for line in lines:
                low = line.lower().replace(':', '').strip()
                if any(low.startswith(k) for k in keywords):
                    if cur["lines"]:
                        sections.append(cur)
                    cur = {"title": line.replace(':', '').strip(), "lines": []}
                elif line:
                    cur["lines"].append(line)
            if cur["lines"]:
                sections.append(cur)
            
            # Создаём структуру с аккордами
            sections_data = []
            chord_idx = 0
            for section in sections:
                sec_data = {"title": section["title"], "lines": []}
                for line_text in section["lines"]:
                    words = line_text.split()
                    # Распределяем аккорды по словам
                    line_chords = []
                    for word in words:
                        chord = chords_seq[chord_idx] if chord_idx < len(chords_seq) else ""
                        line_chords.append({"word": word, "chord": chord})
                        chord_idx += 1
                    sec_data["lines"].append(line_chords)
                sections_data.append(sec_data)
            
            st.session_state.sections_data = sections_data
            st.session_state.song_title = song_title
            st.session_state.processed = True
            st.success("✅ Готово! Редактируй ниже.")
            
        except Exception as e:
            st.error(f"❌ {e}")

# ШАГ 3: Редактор
if st.session_state.processed:
    st.header("2. Редактор")
    st.info("💡 Впиши аккорды для каждого слова (или оставь пустым)")
    
    for si, section in enumerate(st.session_state.sections_data):
        with st.expander(f"🎵 {section['title']}", expanded=True):
            for li, line in enumerate(section["lines"]):
                st.markdown(f"**Строка {li+1}:**")
                cols = st.columns(min(len(line), 6))
                for wi, word_data in enumerate(line):
                    with cols[wi % 6]:
                        chord = st.text_input(
                            f"Аккорд",
                            value=word_data["chord"],
                            key=f"ch_{si}_{li}_{wi}"
                        )
                        st.markdown(f"**{word_data['word']}**")
                        st.session_state.sections_data[si]["lines"][li][wi]["chord"] = chord
                st.markdown("---")
    
    # ШАГ 4: Превью и скачивание
    st.header("3. Превью и скачивание")
    
    if st.button("👁️ Показать превью PDF", type="primary"):
        buf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        c = canvas.Canvas(buf.name, pagesize=A4)
        w, h = A4
        y = h - 2*cm
        
        # Заголовок
        c.setFont(FONT, 16)
        c.setFillColor(colors.black)
        c.drawCentredString(w/2, y, st.session_state.song_title.upper())
        y -= 0.8*cm
        
        # Тональность
        if st.session_state.key_detected:
            c.setFont(FONT, 9)
            c.setFillColor(colors.grey)
            c.drawCentredString(w/2, y, f"Тональность: {st.session_state.key_detected}")
            y -= 0.8*cm
        
        # Секции
        for section in st.session_state.sections_data:
            if y < 3*cm:
                c.showPage()
                y = h - 2*cm
            
            # Заголовок секции
            c.setFont(FONT, 10)
            c.setFillColor(colors.HexColor('#000080'))
            c.drawString(1*cm, y, section["title"] + ":")
            y -= 0.6*cm
            
            # Строки
            for line in section["lines"]:
                if y < 2.5*cm:
                    c.showPage()
                    y = h - 2*cm
                
                # Проверяем есть ли аккорды
                has_chords = any(w["chord"] for w in line)
                
                if has_chords:
                    # Рисуем аккорды ОДНОЙ СТРОКОЙ над текстом
                    c.setFont(FONT, 9)
                    c.setFillColor(colors.HexColor('#00008B'))
                    x = 1*cm
                    for word_data in line:
                        if word_data["chord"]:
                            c.drawString(x, y + 14, word_data["chord"])
                        x += c.stringWidth(word_data["word"] + " ", FONT, 11)
                
                # Текст
                c.setFillColor(colors.black)
                c.setFont(FONT, 11)
                x = 1*cm
                for word_data in line:
                    c.drawString(x, y, word_data["word"])
                    x += c.stringWidth(word_data["word"] + " ", FONT, 11)
                
                y -= 0.7*cm
            
            y -= 0.4*cm
        
        # Подпись
        c.setFillColor(colors.grey)
        c.setFont(FONT, 8)
        c.drawCentredString(w/2, 1*cm, 'ПЕСНИ ПАВЛОВЫХ©')
        c.save()
        
        with open(buf.name, 'rb') as f:
            pdf_bytes = f.read()
        
        # Показываем превью
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
            file_name=f"{st.session_state.song_title}_pavlovy.pdf",
            mime="application/pdf",
            type="primary"
        )
        
        os.unlink(buf.name)
