
import asyncio
import os
from pathlib import Path
from io import BytesIO

from ultralytics import YOLO
from PIL import Image

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile

from dotenv import load_dotenv


# Загружаем токен
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

# Загружаем последнюю модель
MODEL_PATH = Path("runs/detect/train-12/weights/best.pt")

model = YOLO(str(MODEL_PATH))

# Проверяем классы модели
print("Классы модели:", model.names)

# Создаём бота и диспетчер
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# Команда /start
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Привет! 🚢⛰️\n"
        "Отправь мне фотографию, и я найду на ней корабли и горы."
    )


# Обработка фотографии
@dp.message(F.photo)
async def handle_photo(message: Message):

    await message.answer("Обрабатываю фотографию... 🔍")

    # Получаем фотографию
    photo = message.photo[-1]

    file_info = await bot.get_file(photo.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)

    # Открываем изображение
    image = Image.open(downloaded_file).convert("RGB")

    # Предсказание модели
    results = model.predict(image, conf=0.25)

    # Выводим найденные классы в терминал
    print("Найденные классы:", results[0].boxes.cls.tolist())

    # Рисуем рамки без подписей классов
    result_image = results[0].plot(labels=False)

    # Переводим BGR в RGB
    result_image = result_image[:, :, ::-1]

    # Сохраняем изображение в память
    output = BytesIO()
    Image.fromarray(result_image).save(output, format="JPEG")

    # Отправляем результат
    result_file = BufferedInputFile(
        output.getvalue(),
        filename="prediction.jpg"
    )

    await message.answer_photo(photo=result_file)


# Если отправили не фотографию
@dp.message()
async def other_messages(message: Message):
    await message.answer("Отправь мне фотографию для распознавания 😊")


# Запуск бота
async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
