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
import requests

st.set_page_config(page_title="ПЕСНИ ПАВЛОВЫХ", layout="wide")
st.title("🎵 Генератор аккордов и текста")
st.markdown("Загрузи песню — получи текст с аккордами в PDF")

# Скачиваем шрифт
@st.cache_resource
def get_font():
    font_path = "PTSans-Regular.ttf"
    if not os.path.exists(font_path):
        try:
            url = "https://github.com/google/fonts/raw/main/ofl/ptsans/PTSans-Regular.ttf"
            response = requests.get(url, timeout=60)
            if response.status_code == 200 and len(response.content) > 1000:
                with open(font_path, "wb") as f:
                    f.write(response.content)
                return font_path
        except:
            pass
    return font_path if os.path.exists(font_path) else None

font_file = get_font()

uploaded_file = st.file_uploader("Выбери аудиофайл (MP3 до 10MB)", type=['mp3'])

if uploaded_file is not None:
    if st.button("🚀 Обработать"):
        with st.spinner('Обработка... 2-3 мин...'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    audio_path = tmp_file.name
                
                st.info("🎤 Распознаю текст...")
                model = whisper.load_model("base")
                result = model.transcribe(audio_path, language="ru", word_timestamps=True)
                
                words_data = []
                for segment in result["segments"]:
                    for word in segment.get("words", []):
                        words_data.append({"word": word["word"].strip(), "start": word["start"]})
                
                st.info("🎼 Анализирую аккорды...")
                y, sr = librosa.load(audio_path, sr=22050, duration=180)
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                
                chord_templates = {
                    'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    'G': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                    'Am': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    'F': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    'Dm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    'E': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    'Em': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    'A': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    'D': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    'Bm': [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                }
                
                chords_data = []
                hop_length = 512
                time_per_frame = hop_length / sr
                prev_chord = ""
                
                for i in range(0, chroma.shape[1], 10):
                    frame_chroma = chroma[:, i]
                    frame_chroma = frame_chroma / (np.linalg.norm(frame_chroma) + 1e-10)
                    best_chord = ""
                    max_corr = 0.6
                    
                    for chord_name, template in chord_templates.items():
                        template = np.array(template) / (np.linalg.norm(template) + 1e-10)
                        for shift in range(12):
                            shifted_template = np.roll(template, shift)
                            corr = np.dot(frame_chroma, shifted_template)
                            if corr > max_corr:
                                max_corr = corr
                                notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                                root = notes[shift % 12]
                                best_chord = root + chord_name[1:]
                    
                    time_sec = i * time_per_frame
                    if best_chord != prev_chord:
                        chords_data.append({"chord": best_chord, "time": time_sec})
                        prev_chord = best_chord
                
                # Синхронизация
                aligned_words = []
                current_chord_idx = 0
                for w in words_data:
                    while current_chord_idx < len(chords_data) - 1 and chords_data[current_chord_idx + 1]["time"] <= w["start"]:
                        current_chord_idx += 1
                    chord = chords_data[current_chord_idx]["chord"] if chords_data else ""
                    aligned_words.append({
                        "word": w["word"],
                        "chord": chord if (current_chord_idx == 0 or chords_data[current_chord_idx]["time"] >= w["start"] - 0.3) else ""
                    })
                
                # Блоки
                total = len(aligned_words)
                part = total // 3 if total > 0 else 1
                blocks = [
                    {"title": "Куплет 1", "words": aligned_words[:part]},
                    {"title": "Припев", "words": aligned_words[part:part*2]},
                    {"title": "Бридж", "words": aligned_words[part*2:]}
                ]
                
                # Генерация PDF с reportlab
                buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                c = canvas.Canvas(buffer.name, pagesize=A4)
                width, height = A4
                
                # Регистрируем шрифт
                if font_file and os.path.exists(font_file):
                    try:
                        pdfmetrics.registerFont(TTFont('PTSans', font_file))
                        font_name = 'PTSans'
                    except:
                        font_name = 'Helvetica'
                        st.warning("⚠️ Шрифт не загрузился")
                else:
                    font_name = 'Helvetica'
                    st.warning("⚠️ Использую стандартный шрифт")
                
                y_position = height - 2*cm
                
                for block in blocks:
                    # Заголовок блока
                    c.setFont(font_name, 12)
                    c.setFillColor(colors.black)
                    c.drawString(1*cm, y_position, block["title"])
                    y_position -= 0.7*cm
                    
                    # Формируем строки
                    lines = []
                    current_words = []
                    current_chords = []
                    
                    for item in block["words"]:
                        current_words.append(item["word"])
                        if item["chord"]:
                            current_chords.append((len(current_words) - 1, item["chord"]))
                        if len(current_words) >= 7:
                            lines.append({"words": current_words, "chords": current_chords})
                            current_words = []
                            current_chords = []
                    
                    if current_words:
                        lines.append({"words": current_words, "chords": current_chords})
                    
                    # Рисуем строки
                    for line in lines:
                        if y_position < 3*cm:
                            c.showPage()
                            y_position = height - 2*cm
                        
                        # Аккорды
                        if line["chords"]:
                            c.setFont(font_name, 9)
                            c.setFillColor(colors.HexColor('#00008B'))
                            x_pos = 1*cm
                            chord_idx = 0
                            for word_idx, word in enumerate(line["words"]):
                                if chord_idx < len(line["chords"]) and line["chords"][chord_idx][0] == word_idx:
                                    c.drawString(x_pos, y_position + 12, line["chords"][chord_idx][1])
                                    chord_idx += 1
                                x_pos += c.stringWidth(word + " ", font_name, 10)
                        
                        # Текст
                        c.setFillColor(colors.black)
                        c.setFont(font_name, 10)
                        text_line = " ".join(line["words"])
                        c.drawString(1*cm, y_position, text_line)
                        y_position -= 0.6*cm
                    
                    y_position -= 0.5*cm
                
                # Подпись
                c.setFillColor(colors.grey)
                c.setFont(font_name, 9)
                c.drawCentredString(width/2, 1*cm, 'ПЕСНИ ПАВЛОВЫХ©')
                
                c.save()
                
                # Читаем PDF
                with open(buffer.name, 'rb') as f:
                    pdf_data = f.read()
                
                os.unlink(buffer.name)
                os.unlink(audio_path)
                
                st.success("✅ Готово!")
                st.download_button(
                    label="📥 Скачать PDF",
                    data=pdf_data,
                    file_name="song_pavlovy.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
