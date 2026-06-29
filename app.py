import streamlit as st
import whisper
import librosa
import numpy as np
import tempfile
import os
from io import BytesIO
from fpdf import FPDF
import requests

st.set_page_config(page_title="ПЕСНИ ПАВЛОВЫХ", layout="wide")

st.title("🎵 Генератор аккордов и текста")
st.markdown("Загрузи песню — получи текст с аккордами в PDF")

# Скачивание шрифтов при запуске
@st.cache_resource
def get_fonts():
    if not os.path.exists("DejaVuSans.ttf"):
        try:
            r = requests.get("https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf", timeout=10)
            with open("DejaVuSans.ttf", "wb") as f:
                f.write(r.content)
        except:
            # Если не получилось, создаём пустой файл
            with open("DejaVuSans.ttf", "wb") as f:
                f.write(b"")
    return "DejaVuSans.ttf"

font_path = get_fonts()

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
                        words_data.append({
                            "word": word["word"].strip(),
                            "start": word["start"]
                        })
                
                st.info("🎼 Анализирую аккорды...")
                y, sr = librosa.load(audio_path, sr=22050, duration=180)  # Ограничиваем 3 минуты
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                
                chord_templates = {
                    'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
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
                        "chord": chord if (current_chord_idx == 0 or (current_chord_idx > 0 and chords_data[current_chord_idx]["time"] >= w["start"] - 0.3)) else ""
                    })
                
                # Разделение на блоки
                total = len(aligned_words)
                part = total // 3 if total > 0 else 1
                blocks = [
                    {"title": "Куплет 1", "words": aligned_words[:part]},
                    {"title": "Припев", "words": aligned_words[part:part*2]},
                    {"title": "Бридж", "words": aligned_words[part*2:]}
                ]
                
                # Генерация PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=20)
                
                # Пытаемся добавить шрифт
                try:
                    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
                    pdf.set_font('DejaVu', '', 11)
                except:
                    # Если шрифт не загрузился, используем стандартный (кириллица не будет работать)
                    pdf.set_font('Arial', '', 11)
                    st.warning("⚠️ Шрифт не загрузился. Кириллица может отображаться некорректно.")
                
                for block in blocks:
                    # Заголовок блока
                    pdf.set_font('DejaVu', 'B', 12)
                    pdf.cell(0, 10, block["title"], ln=True)
                    pdf.ln(3)
                    
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
                        if line["chords"]:
                            pdf.set_font('DejaVu', '', 9)
                            pdf.set_text_color(0, 0, 128)
                            chord_str = ""
                            last_pos = -1
                            for idx, chord in line["chords"]:
                                while last_pos + 1 < idx:
                                    chord_str += "    "
                                    last_pos += 1
                                chord_str += chord + "  "
                                last_pos = idx
                            pdf.cell(0, 5, chord_str, ln=True)
                        
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font('DejaVu', '', 10)
                        pdf.cell(0, 6, " ".join(line["words"]), ln=True)
                        pdf.ln(1)
                    
                    pdf.ln(5)
                
                # Подпись
                pdf.set_y(-15)
                pdf.set_font('DejaVu', 'B', 9)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 10, 'ПЕСНИ ПАВЛОВЫХ©', align='C')
                
                pdf_output = pdf.output(dest='S').encode('latin-1', errors='replace')
                
                st.success("✅ Готово!")
                st.download_button(
                    label="📥 Скачать PDF",
                    data=pdf_output,
                    file_name="song_pavlovy.pdf",
                    mime="application/pdf"
                )
                
                os.unlink(audio_path)
                
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
