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

# Регистрация шрифта с кириллицей
FONT = 'Helvetica'
try:
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    FONT = 'DejaVu'
except:
    pass

st.set_page_config(page_title="ПЕСНИ ПАВЛОВЫХ", layout="wide", page_icon="🎵")
st.title("🎵 ПЕСНИ ПАВЛОВЫХ — Авто-генератор аккордов")
st.markdown("Загрузи MP3 → получи PDF с аккордами над словами как в образцах")

# ---------- ЗАГРУЗКА ----------
uploaded_audio = st.file_uploader("📁 Выбери MP3 файл", type=['mp3'])
song_title = st.text_input("Название песни", "Без названия")

if uploaded_audio is not None:
    st.audio(uploaded_audio)
    
    if st.button("🚀 Обработать автоматически", type="primary"):
        with st.spinner("⏳ Обрабатываю... 3-5 минут..."):
            try:
                # Сохраняем аудио
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                    tmp.write(uploaded_audio.read())
                    audio_path = tmp.name
                
                # 1. Распознаём текст с таймкодами слов
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
                
                # 2. Анализируем аккорды с таймкодами
                st.info("🎼 Анализирую гармонию...")
                y, sr = librosa.load(audio_path, sr=22050, duration=240)
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                
                # Шаблоны аккордов
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
                st.info(" Определяю тональность...")
                chord_counts = {}
                for ct in chords_timeline:
                    chord_counts[ct["chord"]] = chord_counts.get(ct["chord"], 0) + 1
                
                # Тональность = самый частый мажорный аккорд
                major_chords = [c for c in chord_counts.keys() if not c.endswith('m') and not c.endswith('7')]
                if major_chords:
                    tonic = max(major_chords, key=lambda x: chord_counts[x])
                    key_name = f"{tonic} (мажор)"
                else:
                    tonic = max(chord_counts.keys(), key=lambda x: chord_counts[x])
                    key_name = f"{tonic} (минор)"
                
                # 4. Привязываем аккорды к словам
                st.info("📝 Синхронизирую аккорды со словами...")
                words_with_chords = []
                prev_chord_for_word = ""
                
                for word in words_with_time:
                    # Находим аккорд на момент начала слова
                    chord_at_word = ""
                    for ct in reversed(chords_timeline):
                        if ct["time"] <= word["start"] + 0.1:
                            chord_at_word = ct["chord"]
                            break
                    
                    # Показываем аккорд только если он сменился
                    display_chord = chord_at_word if chord_at_word != prev_chord_for_word else ""
                    prev_chord_for_word = chord_at_word
                    
                    words_with_chords.append({
                        "text": word["text"],
                        "chord": display_chord
                    })
                
                # 5. Разделяем на секции (эвристика)
                total_words = len(words_with_chords)
                part = max(1, total_words // 4)
                
                sections = [
                    {"title": "Куплет 1", "words": words_with_chords[:part]},
                    {"title": "Куплет 2", "words": words_with_chords[part:part*2]},
                    {"title": "Припев", "words": words_with_chords[part*2:part*3]},
                    {"title": "Бридж", "words": words_with_chords[part*3:]}
                ]
                
                # 6. Генерируем PDF как в образцах
                buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                c = canvas.Canvas(buffer.name, pagesize=A4)
                w, h = A4
                y = h - 2*cm
                
                # Заголовок песни (крупно)
                c.setFont(FONT, 20)
                c.setFillColor(colors.black)
                c.drawCentredString(w/2, y, song_title.upper())
                y -= 1.2*cm
                
                # Тональность и темп
                c.setFont(FONT, 11)
                c.setFillColor(colors.HexColor('#333333'))
                c.drawCentredString(w/2, y, f"Тональность: {key_name} | Темп: ~120 BPM")
                y -= 1.5*cm
                
                # Секции
                for section in sections:
                    if y < 4*cm:
                        c.showPage()
                        y = h - 2*cm
                    
                    # Заголовок секции
                    c.setFont(FONT, 13)
                    c.setFillColor(colors.HexColor('#000080'))
                    c.drawString(1*cm, y, section["title"] + ":")
                    y -= 0.8*cm
                    
                    # Формируем строки по 5-6 слов
                    words = section["words"]
                    line_size = 6
                    for i in range(0, len(words), line_size):
                        if y < 3*cm:
                            c.showPage()
                            y = h - 2*cm
                        
                        line_words = words[i:i+line_size]
                        
                        # Аккорды над словами
                        has_chords = any(w["chord"] for w in line_words)
                        if has_chords:
                            c.setFont(FONT, 11)
                            c.setFillColor(colors.HexColor('#00008B'))
                            x = 1.2*cm
                            for word_data in line_words:
                                if word_data["chord"]:
                                    c.drawString(x, y + 14, word_data["chord"])
                                x += c.stringWidth(word_data["text"] + " ", FONT, 13)
                        
                        # Текст строки (крупно)
                        c.setFillColor(colors.black)
                        c.setFont(FONT, 13)
                        text_line = " ".join(w["text"] for w in line_words)
                        c.drawString(1.2*cm, y, text_line)
                        y -= 0.8*cm
                    
                    y -= 0.5*cm
                
                # Подпись
                c.setFillColor(colors.grey)
                c.setFont(FONT, 9)
                c.drawCentredString(w/2, 1*cm, 'ПЕСНИ ПАВЛОВЫХ©')
                c.save()
                
                with open(buffer.name, 'rb') as f:
                    pdf_data = f.read()
                
                os.unlink(buffer.name)
                os.unlink(audio_path)
                
                st.success("✅ PDF готов!")
                st.download_button(
                    label="📥 Скачать PDF",
                    data=pdf_data,
                    file_name=f"{song_title}_pavlovy.pdf",
                    mime="application/pdf"
                )
                
                # Показываем превью
                st.markdown(f"###  {song_title}")
                st.markdown(f"**Тональность:** {key_name}")
                for section in sections:
                    st.markdown(f"**{section['title']}:**")
                    text = " ".join(w["text"] for w in section["words"])
                    st.write(text)
                
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
                import traceback
                st.error(traceback.format_exc())
