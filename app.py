# Вместо скачивания шрифта, используем встроенный
# Reportlab имеет встроенные шрифты с кириллицей
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

# Встроенный шрифт DejaVu (уже есть в системе)
try:
    # Пробуем зарегистрировать системный шрифт
    pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    font_name = 'DejaVuSans'
except:
    font_name = 'Helvetica'  # fallback
    st.warning("⚠️ Использую стандартный шрифт")
