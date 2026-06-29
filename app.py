import streamlit as st
import whisper
import librosa
import numpy as np
import tempfile
import os
from io import BytesIO
from fpdf import FPDF
import requests

# Скачиваем шрифт с поддержкой кириллицы
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/dejavusans/DejaVuSans.ttf"
FONT_BOLD_URL = "https://github.com/google/fonts/raw/main/ofl/dejavusans/DejaVuSans-Bold.ttf"

def download_fonts():
    if not os.path.exists("DejaVuSans.ttf"):
        r = requests.get(FONT_URL)
        with open("DejaVuSans.ttf", "wb") as f:
            f.write(r.content)
    if not os.path.exists("DejaVuSans-Bold.ttf"):
        r = requests.get(FONT_BOLD_URL)
        with open("DejaVuSans-Bold.ttf", "wb") as f:
            f.write(r.content)

download_fonts()

st.set_page_config(page_title="ПЕСНИ ПАВЛОВЫХ - Аккорды и Текст", layout="wide")

st.title("🎵 Генератор аккордов и текста")
st.markdown("### Загрузи песню — получи текст с аккордами в PDF")

st.sidebar.header("Настройки")
model_size = st.sidebar.selectbox("Модель распознавания", ["base", "small"])
confidence_threshold = st.sidebar.slider("Порог уверенности аккордов", 0.3, 0.9, 0.5)

uploaded_file = st.file_uploader(" Выбери аудиофайл (MP3, WAV, M4A)", type=['mp3', 'wav', 'm4a', 'ogg'])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button(" Обработать песню"):
        with st.spinner('⏳ Обрабатываю... Это займёт 2-5 минут...'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    audio_path = tmp_file.name
                
                progress_bar = st.progress(0)
                st.info("🎤 Распознаю текст песни...")
                model = whisper.load_model(model_size)
                result = model.transcribe(audio_path, language="ru", word_timestamps=True)
                progress_bar.progress(33)
                
                words_data = []
                for segment in result["segments"]:
                    for word in segment.get("words", []):
                        words_data.append({
                            "word": word["word"].strip(),
                            "start": word["start"],
                            "end": word["end"]
                        })
                
                st.info("🎼 Анализирую аккорды...")
                y, sr = librosa.load(audio_path, sr=22050)
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                progress_bar.progress(66)
                
                chord_templates = {
                    'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    'G': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                    'Gm': [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    'Am': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    'F': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    'Dm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    'E': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    'Em': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    'A': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    'D': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    'Bm': [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    'Bb': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    'F#': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    'C#': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                }
                
                chords_data = []
                hop_length = 512
                time_per_frame = hop_length / sr
                prev_chord = "N.C."
                
                for i in range(chroma.shape[1]):
                    frame_chroma = chroma[:, i]
                    frame_chroma = frame_chroma / (np.linalg.norm(frame_chroma) + 1e-10)
                    
                    best_chord = "N.C."
                    max_corr = confidence_threshold
                    
                    for chord_name, template in chord_templates.items():
                        template = np.array(template) / (np.linalg.norm(template) + 1e-10)
                        for shift in range(12):
                            shifted_template = np.roll(template, shift)
                            corr = np.dot(frame_chroma, shifted_template)
                            if corr > max_corr:
                                max_corr = corr
                                notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                                root = notes[shift % 12]
                                suffix = chord_name[1:] if len(chord_name) > 1 else ''
                                best_chord = root + suffix
                    
                    time_sec = i * time_per_frame
                    if best_chord != prev_chord:
                        chords_data.append({"chord": best_chord, "time": time_sec})
                        prev_chord = best_chord
                
                progress_bar.progress(90)
                
                st.info("📝 Синхронизирую аккорды с текстом...")
                aligned_words = []
                current_chord_idx = 0
                
                for w in words_data:
                    while current_chord_idx < len(chords_data) - 1 and chords_data[current_chord_idx + 1]["time"] <= w["start"]:
                        current_chord_idx += 1
                    
                    chord = chords_data[current_chord_idx]["chord"] if chords_data else "N.C."
                    display_chord = chord if (current_chord_idx == 0 or (current_chord_idx > 0 and chords_data[current_chord_idx]["time"] >= w["start"] - 0.2)) else ""
                    if w == words_data[0]:
                        display_chord = chord
                    
                    aligned_words.append({"word": w["word"], "chord": display_chord, "time": w["start"]})
                
                total_words = len(aligned_words)
                part_size = total_words // 3
                blocks = [
                    {"type": "verse", "title": "Куплет 1", "words": aligned_words[:part_size]},
                    {"type": "chorus", "title": "Припев", "words": aligned_words[part_size:part_size*2]},
                    {"type": "bridge", "title": "Бридж", "words": aligned_words[part_size*2:]}
                ]
                
                progress_bar.progress(100)
                st.success("✅ Готово!")
                
                # Генерация PDF в формате как в референсах
                class PDF(FPDF):
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('DejaVu', 'B', 9)
                        self.set_text_color(100, 100, 100)
                        self.cell(0, 10, 'ПЕСНИ ПАВЛОВЫХ\u00A9', align='C')
                
                pdf = PDF()
                pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
                pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
                pdf.set_auto_page_break(auto=True, margin=20)
                pdf.add_page()
                
                # Заголовок песни (если нужно)
                # pdf.set_font('DejaVu', 'B', 16)
                # pdf.cell(0, 15, 'Название песни', align='C', ln=True)
                # pdf.ln(5)
                
                for block in blocks:
                    # Заголовок блока жирным
                    pdf.set_font('DejaVu', 'B', 12)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 10, block["title"], ln=True)
                    pdf.ln(3)
                    
                    # Формируем строки с аккордами и текстом
                    # Собираем все слова и аккорды в линии
                    lines = []
                    current_line_words = []
                    current_line_chords = []
                    max_words_per_line = 8  # Примерно 8 слов в строке
                    
                    for item in block["words"]:
                        current_line_words.append(item["word"])
                        if item["chord"] and item["chord"] != "N.C.":
                            current_line_chords.append((len(current_line_words) - 1, item["chord"]))
                        
                        if len(current_line_words) >= max_words_per_line:
                            lines.append({
                                "words": current_line_words,
                                "chords": current_line_chords
                            })
                            current_line_words = []
                            current_line_chords = []
                    
                    # Последняя строка
                    if current_line_words:
                        lines.append({
                            "words": current_line_words,
                            "chords": current_line_chords
                        })
                    
                    # Рисуем каждую строку
                    for line in lines:
                        # Сначала рисуем аккорды над словами
                        if line["chords"]:
                            pdf.set_font('DejaVu', '', 10)
                            pdf.set_text_color(0, 0, 139)  # Тёмно-синий для аккордов
                            
                            chord_text = ""
                            last_pos = -1
                            for word_idx, chord in line["chords"]:
                                # Добавляем пробелы до позиции аккорда
                                while last_pos + 1 < word_idx:
                                    chord_text += "    "
                                    last_pos += 1
                                chord_text += chord + "  "
                                last_pos = word_idx
                            
                            pdf.cell(0, 6, chord_text, ln=True)
                        
                        # Теперь рисуем текст
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font('DejaVu', '', 11)
                        line_text = " ".join(line["words"])
                        pdf.cell(0, 7, line_text, ln=True)
                        pdf.ln(2)
                    
                    pdf.ln(5)
                
                pdf_output = pdf.output(dest='S').encode('latin-1')
                
                st.markdown("### 📄 Предпросмотр:")
                for block in blocks:
                    with st.expander(f"{block['title']}", expanded=True):
                        text_with_chords = ""
                        for item in block["words"]:
                            if item["chord"] and item["chord"] != "N.C.":
                                text_with_chords += f"**{item['chord']}** "
                            text_with_chords += item["word"] + " "
                        st.markdown(text_with_chords)
                
                st.download_button(
                    label="📥 СКАЧАТЬ PDF",
                    data=pdf_output,
                    file_name="song_chords_pavlovy.pdf",
                    mime="application/pdf"
                )
                
                os.unlink(audio_path)
                
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
