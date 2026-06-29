import streamlit as st
import whisper
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

# ---------- ИНИЦИАЛИЗАЦИЯ SESSION STATE ----------
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'sections_data' not in st.session_state:
    st.session_state.sections_data = []
if 'song_title' not in st.session_state:
    st.session_state.song_title = ""
if 'key_detected' not in st.session_state:
    st.session_state.key_detected = ""

# ---------- ШАГ 1: ЗАГРУЗКА ----------
st.header("1. Загрузка файлов")
col1, col2 = st.columns(2)

with col1:
    uploaded_audio = st.file_uploader("🎵 Аудиофайл (MP3)", type=['mp3'])
    song_title = st.text_input("Название песни", "Без названия")

with col2:
    default_text = """Куплет 1:
Кто-то парус поднимет выше,
Кто-то держит в руках штурвал,
Каждый здесь Божий зов услышал,
Он нас Сам в этот путь призвал.

Припев:
Не теряя цели,
Обходя шторма и мели,
Сохраняя веру,
Держи курс на небо!

Куплет 2:
Курс на вечность – одна задача,
Мир меняется в играх волн.

Бридж:
Не смотри в глубину, там холод,
Там вершины подводных скал."""
    
    raw_text = st.text_area("📝 Текст песни (с разделами):", value=default_text, height=300,
                            help="Разделы: Куплет 1:, Припев:, Бридж:, Проигрыш:, Кода:")

# ---------- ШАГ 2: АНАЛИЗ ----------
st.header("2. Автоматический анализ")

if st.button("🎼 Проанализировать и расставить аккорды", type="primary", disabled=not uploaded_audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name
    
    with st.spinner("⏳ Анализирую... 3-5 минут..."):
        try:
            # 1. Распознаём текст с таймкодами
            st.info("🎤 Распознаю текст с таймкодами...")
            model = whisper.load_model("base")
            result = whisper.transcribe(model, audio_path, language="ru", word_timestamps=True)
            
            words_with_time = []
            for segment in result["segments"]:
                for word in segment.get("words", []):
                    words_with_time.append({
                        "text": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"]
                    })
            
            # 2. Анализируем аккорды
            st.info("🎼 Определяю аккорды...")
            y, sr = librosa.load(audio_path, sr=22050, duration=240)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            chord_templates = {
                '': [1,0,0,0,1,0,0,1,0,0,0,0], 'm': [1,0,0,1,0,0,0,1,0,0,0,0],
                '7': [1,0,0,0,1,0,0,1,0,0,1,0], 'm7': [1,0,0,1,0,0,0,1,0,0,1,0],
                'sus4': [1,0,0,0,0,1,0,1,0,0,0,0], 'sus2': [1,0,1,0,0,0,0,1,0,0,0,0],
                '6': [1,0,0,0,1,0,0,1,0,1,0,0], 'maj7': [1,0,0,0,1,0,0,1,0,0,0,1],
            }
            notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            
            chords_timeline = []
            prev_chord = ""
            hop_length = 2048
            time_per_frame = hop_length / sr
            
            for i in range(chroma.shape[1]):
                fc = chroma[:, i]
                fc = fc / (np.linalg.norm(fc) + 1e-10)
                best, mx = "", 0.65
                
                for suf, tpl in chord_templates.items():
                    tpl = np.array(tpl) / (np.linalg.norm(tpl) + 1e-10)
                    for s in range(12):
                        c = np.dot(fc, np.roll(tpl, s))
                        if c > mx:
                            mx = c
                            best = notes[s] + suf
                
                time_sec = i * time_per_frame
                if best and best != prev_chord:
                    chords_timeline.append({"time": time_sec, "chord": best})
                    prev_chord = best
            
            # 3. Определяем тональность
            chord_counts = {}
            for ct in chords_timeline:
                chord_counts[ct["chord"]] = chord_counts.get(ct["chord"], 0) + 1
            
            major_chords = [c for c in chord_counts.keys() if not c.endswith('m') and not c.endswith('7')]
            if major_chords:
                tonic = max(major_chords, key=lambda x: chord_counts[x])
                key_name = f"{tonic} (мажор)"
            else:
                tonic = max(chord_counts.keys(), key=lambda x: chord_counts[x])
                key_name = f"{tonic} (минор)"
            
            st.session_state.key_detected = key_name
            
            # 4. Парсим текст на секции
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
            
            # 5. Привязываем аккорды к словам в каждой строке
            st.info(" Синхронизирую аккорды со словами...")
            sections_with_chords = []
            
            for section in sections:
                section_data = {"title": section["title"], "lines": []}
                
                for line_text in section["lines"]:
                    # Разбиваем строку на слова
                    words_in_line = line_text.split()
                    line_chords = []
                    
                    # Для каждого слова ищем аккорд
                    for word in words_in_line:
                        # Находим ближайший аккорд по времени (упрощённо)
                        chord_for_word = ""
                        # Используем распознанные слова из Whisper для таймкодов
                        for wwt in words_with_time:
                            if wwt["text"].lower() == word.lower():
                                # Нашли слово, ищем аккорд
                                for ct in reversed(chords_timeline):
                                    if ct["time"] <= wwt["start"] + 0.2:
                                        chord_for_word = ct["chord"]
                                        break
                                break
                        
                        line_chords.append({"word": word, "chord": chord_for_word})
                    
                    section_data["lines"].append(line_chords)
                
                sections_with_chords.append(section_data)
            
            st.session_state.sections_data = sections_with_chords
            st.session_state.song_title = song_title
            st.session_state.processed = True
            
            st.success("✅ Анализ завершён! Переходи к редактору ниже.")
            
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            import traceback
            st.error(traceback.format_exc())

# ---------- ШАГ 3: РЕДАКТОР С ПРЕВЬЮ ----------
if st.session_state.processed:
    st.header("3. Редактор (проверь и исправь)")
    
    st.info("💡 **Инструкция:** Для каждой строки редактируй аккорды над словами. Оставь поле пустым если аккорда нет.")
    
    # Редактирование
    for si, section in enumerate(st.session_state.sections_data):
        with st.expander(f"🎵 {section['title']}", expanded=True):
            for li, line in enumerate(section["lines"]):
                st.markdown(f"**Строка {li + 1}:**")
                
                # Показываем слова и поля для аккордов
                cols = st.columns(len(line))
                for wi, word_data in enumerate(line):
                    with cols[wi]:
                        # Поле для аккорда
                        chord_val = st.text_input(
                            "Аккорд",
                            value=word_data["chord"],
                            key=f"ch_{si}_{li}_{wi}",
                            placeholder="E"
                        )
                        # Текст слова (только для отображения)
                        st.markdown(f"**{word_data['word']}**")
                        
                        # Обновляем данные
                        st.session_state.sections_data[si]["lines"][li][wi]["chord"] = chord_val
                
                st.markdown("---")
    
    # ---------- ШАГ 4: ПРЕВЬЮ PDF ----------
    st.header("4. Предварительный просмотр PDF")
    
    if st.button("👁️ Показать превью PDF", type="primary"):
        # Генерируем PDF во временный файл
        buf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        c = canvas.Canvas(buf.name, pagesize=A4)
        w, h = A4
        y = h - 2.5*cm  # Увеличенный отступ сверху
        
        # Заголовок песни (крупно)
        c.setFont(FONT, 22)
        c.setFillColor(colors.black)
        c.drawCentredString(w/2, y, st.session_state.song_title.upper())
        y -= 1.5*cm
        
        # Тональность
        if st.session_state.key_detected:
            c.setFont(FONT, 11)
            c.setFillColor(colors.HexColor('#555555'))
            c.drawCentredString(w/2, y, f"Тональность: {st.session_state.key_detected}")
            y -= 1.5*cm
        
        # Секции
        for section in st.session_state.sections_data:
            if y < 4*cm:
                c.showPage()
                y = h - 2.5*cm
            
            # Заголовок секции
            c.setFont(FONT, 14)
            c.setFillColor(colors.HexColor('#000080'))
            c.drawString(1.5*cm, y, section["title"] + ":")
            y -= 1.2*cm  # Увеличенный отступ после заголовка
            
            # Строки
            for line in section["lines"]:
                if y < 3.5*cm:
                    c.showPage()
                    y = h - 2.5*cm
                
                # Проверяем есть ли аккорды
                has_chords = any(w["chord"] for w in line)
                
                if has_chords:
                    # Рисуем аккорды над словами
                    c.setFont(FONT, 12)
                    c.setFillColor(colors.HexColor('#00008B'))
                    x = 1.5*cm
                    for word_data in line:
                        if word_data["chord"]:
                            c.drawString(x, y + 18, word_data["chord"])  # Аккорд выше
                        x += c.stringWidth(word_data["word"] + " ", FONT, 14)
                
                # Рисуем текст (крупно)
                c.setFillColor(colors.black)
                c.setFont(FONT, 14)
                x = 1.5*cm
                for word_data in line:
                    c.drawString(x, y, word_data["word"])
                    x += c.stringWidth(word_data["word"] + " ", FONT, 14)
                
                y -= 1.2*cm  # Увеличенное расстояние между строками
            
            y -= 0.8*cm  # Отступ между секциями
        
        # Подпись
        c.setFillColor(colors.grey)
        c.setFont(FONT, 10)
        c.drawCentredString(w/2, 1.2*cm, 'ПЕСНИ ПАВЛОВЫХ©')
        c.save()
        
        # Показываем PDF
        with open(buf.name, 'rb') as f:
            pdf_bytes = f.read()
        
        st.markdown("### 📄 Предварительный просмотр:")
        st.components.v1.html(
            f'<iframe src="data:application/pdf;base64,{pdf_bytes.hex()}" width="100%" height="800px"></iframe>',
            height=800
        )
        
        st.success("✅ Превью готово! Если всё правильно — скачай PDF ниже.")
        
        # Кнопка скачивания
        st.download_button(
            label="📥 Скачать PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state.song_title}_pavlovy.pdf",
            mime="application/pdf",
            type="primary"
        )
        
        os.unlink(buf.name)
