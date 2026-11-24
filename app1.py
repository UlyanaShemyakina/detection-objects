import streamlit as st
from ultralytics import YOLO
from PIL import Image

import pillow_avif

# Заголовок приложения
st.title("🚢⛰️ Детектор кораблей и гор")

# st.write("Загрузите фотографию, и модель определит, есть ли на ней корабль или гора.")

# Загружаем модель
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train-12/weights/best.pt")

model = load_model()

# Показываем классы модели
# st.write("Классы модели:", model.names)

# Загрузка фотографии
uploaded_file = st.file_uploader(
    "Загрузите фото",
    type=["jpg", "jpeg", "png", "avif", "webp"]
)

if uploaded_file is not None:

    # Открываем изображение
    image = Image.open(uploaded_file).convert("RGB")

    # Создаём место для фотографии
    image_placeholder = st.empty()

    # Показываем загруженную фотографию
    image_placeholder.image(
        image,
        caption="Загруженная фотография",
        use_container_width=True
    )

    if st.button("Найти объекты"):

        results = model.predict(image, conf=0.25)

        # Рисуем bounding box
        result_image = results[0].plot()

        # Заменяем исходную фотографию результатом
        image_placeholder.image(
            result_image,
            caption="Результат детекции",
            channels="BGR",
            use_container_width=True
        )

        # # Получаем обнаруженные классы
        # if len(results[0].boxes) > 0:

        #     detected_classes = set(
        #         model.names[int(cls)]
        #         for cls in results[0].boxes.cls
        #     )

        #     st.success("Обнаружены объекты: " + ", ".join(detected_classes))

        # else:
        #     st.warning("Объекты не обнаружены.")