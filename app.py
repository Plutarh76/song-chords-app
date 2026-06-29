import streamlit as st
import whisper
import librosa
import numpy as np
import tempfile
import os
from fpdf import FPDF
import requests
import base64

st.set_page_config(page_title="ПЕСНИ ПАВЛОВЫХ", layout="wide")

st.title("🎵 Генератор аккордов и текста")
st.markdown("Загрузи песню — получи текст с аккордами в PDF")

# Встроенный шрифт с поддержкой кириллицы (base64)
# Используем бесплатный шрифт PT Sans от ParaType
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/ptsans/PTSans-Regular.ttf"
FONT_BOLD_URL = "https://github.com/google/fonts/raw/main/ofl/ptsans/PTSans-Bold.ttf"

def download_fonts():
    """Скачиваем шрифты с поддержкой кириллицы"""
    try:
        if not os.path.exists("PTSans-Regular.ttf"):
            r = requests.get(FONT_URL, timeout=30)
            with open("PTSans-Regular.ttf", "wb") as f:
                f.write(r.content)
        
        if not os.path.exists("PTSans-Bold.ttf"):
            r = requests.get(FONT_BOLD_URL, timeout=30)
            with open("PTSans-Bold.ttf", "wb") as f:
                f.write(r.content)
        
        return True
    except Exception as e:
        st.warning(f"⚠️ Шрифты не загрузились: {e}")
        return False

fonts_ready = download_fonts()

uploaded_file = st.file_uploader("Выбери аудиофайл (MP3 до 10MB)", type=['mp3'])

if uploaded_file is not None:
    if st.button("🚀 Обработать"):
        with st.spinner('Обработка... 2-3 мин...'):
            try:
                # Сохраняем файл
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    audio_path = tmp_file.name
                
                # 1. Распознавание текста
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
                
                # 2. Анализ аккордов
                st.info("🎼 Анализирую аккорды...")
                y, sr = librosa.load(audio_path, sr=22050, duration=180)
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                
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
                    'Eb': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    'F#': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    'C#': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    'C#m': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    'G#m': [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    'F#m': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                }
                
                chords_data = []
                hop_length = 512
                time_per_frame = hop_length / sr
                prev_chord = ""
                
                for i in range(0, chroma.shape[1], 10):
                    frame_chroma = chroma[:, i]
                    frame_chroma = frame_chroma / (np.linalg.norm(frame_chroma) + 1e-10)
                    
                    best_chord = ""
                    max_corr = 0.55
                    
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
                
                # 3. Синхронизация аккордов и слов
                st.info("📝 Синхронизирую...")
                aligned_words = []
                current_chord_idx = 0
                
                for w in words_data:
                    while current_chord_idx < len(chords_data) - 1 and chords_data[current_chord_idx + 1]["time"] <= w["start"]:
                        current_chord_idx += 1
                    
                    chord = chords_data[current_chord_idx]["chord"] if chords_data else ""
                    display_chord = chord if (current_chord_idx == 0 or (current_chord_idx > 0 and chords_data[current_chord_idx]["time"] >= w["start"] - 0.3)) else ""
                    
                    aligned_words.append({
                        "word": w["word"], 
                        "chord": display_chord
                    })
                
                # 4. Разделение на блоки
                total = len(aligned_words)
                part = total // 3 if total > 0 else 1
                blocks = [
                    {"title": "Куплет 1", "words": aligned_words[:part]},
                    {"title": "Припев", "words": aligned_words[part:part*2]},
                    {"title": "Бридж", "words": aligned_words[part*2:]}
                ]
                
                # 5. Генерация PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=20)
                
                # Добавляем шрифты
                font_regular = 'PTSans-Regular.ttf'
                font_bold = 'PTSans-Bold.ttf'
                
                if os.path.exists(font_regular) and os.path.exists(font_bold):
                    pdf.add_font('PTSans', '', font_regular, uni=True)
                    pdf.add_font('PTSans', 'B', font_bold, uni=True)
                    main_font = 'PTSans'
                else:
                    # Fallback
                    main_font = 'Helvetica'
                    st.warning("⚠️ Использую стандартный шрифт")
                
                # Обрабатываем каждый блок
                for block in blocks:
                    # Заголовок блока жирным
                    pdf.set_font(main_font, 'B', 12)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 10, block["title"], ln=True)
                    pdf.ln(3)
                    
                    # Формируем строки с аккордами
                    lines = []
                    current_words = []
                    current_chords = []
                    
                    for item in block["words"]:
                        current_words.append(item["word"])
                        if item["chord"]:
                            current_chords.append((len(current_words) - 1, item["chord"]))
                        
                        # Разбиваем на строки по 6-8 слов
                        if len(current_words) >= 7:
                            lines.append({"words": current_words, "chords": current_chords})
                            current_words = []
                            current_chords = []
                    
                    # Последняя строка
                    if current_words:
                        lines.append({"words": current_words, "chords": current_chords})
                    
                    # Рисуем каждую строку
                    for line in lines:
                        # Сначала аккорды (если есть)
                        if line["chords"]:
                            pdf.set_font(main_font, '', 10)
                            pdf.set_text_color(0, 0, 139)  # Тёмно-синий
                            
                            # Формируем строку с аккордами
                            chord_line = ""
                            last_idx = -1
                            for word_idx, chord in line["chords"]:
                                # Добавляем пробелы до нужной позиции
                                while last_idx + 1 < word_idx:
                                    chord_line += "     "
                                    last_idx += 1
                                chord_line += chord + "   "
                                last_idx = word_idx
                            
                            pdf.cell(0, 6, chord_line, ln=True)
                        
                        # Теперь текст песни
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font(main_font, '', 11)
                        text_line = " ".join(line["words"])
                        pdf.cell(0, 7, text_line, ln=True)
                        pdf.ln(2)
                    
                    pdf.ln(5)
                
                # Подпись внизу страницы
                pdf.set_y(-15)
                pdf.set_font(main_font, 'B', 9)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 10, 'ПЕСНИ ПАВЛОВЫХ©', align='C')
                
                # Сохраняем PDF
                pdf_output = pdf.output(dest='S').encode('latin-1', errors='replace')
                
                st.success("✅ Готово!")
                
                # Кнопка скачивания
                st.download_button(
                    label="📥 Скачать PDF",
                    data=pdf_output,
                    file_name="song_pavlovy.pdf",
                    mime="application/pdf"
                )
                
                # Очищаем временный файл
                os.unlink(audio_path)
                
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
