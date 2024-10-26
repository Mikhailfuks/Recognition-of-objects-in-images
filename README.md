#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Функция для получения названий выходных слоев модели
vector<String> getOutputsNames(const Net& net) {
    static vector<String> names;
    if (names.empty()) {
        // Получаем все имена слоев
        vector<String> layers = net.getLayerNames();

        // Извлекаем имена выходных слоев
        for (size_t i = 0; i < layers.size(); ++i) {
            // Используем метод getUnconnectedOutLayersNames для определения выходных слоев
            if (net.getLayer(layers[i])->type == "Region" ||
                net.getLayer(layers[i])->type == "DetectionOutput") {
                names.push_back(layers[i]);
            }
        }
    }
    return names;
}

int main() {
    // Загрузка модели YOLOv3
    String modelConfiguration = "yolov3.cfg"; // Путь к файлу конфигурации YOLOv3
    String modelWeights = "yolov3.weights"; // Путь к файлу весов YOLOv3
    Net net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);

    // Загрузка изображения
    Mat image = imread("image.jpg"); // Замените "image.jpg" на путь к вашему изображению

    // Проверка, загружено ли изображение
    if (image.empty()) {
        cerr << "Ошибка загрузки изображения!" << endl;
        return -1;
    }

    // Преобразование изображения в формат BGR
    Mat blob;
    dnn::blobFromImage(image, blob, 1.0 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);

    // Задание входных данных модели
    net.setInput(blob);

    // Выполнение предсказания
    vector<String> outNames = getOutputsNames(net);
    vector<Mat> outs;
    net.forward(outs, outNames);

    // Обработка результатов
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i) {
        // Перебираем все обнаруженные объекты
        float* data = (float*)outs[i].data;
        for (size_t j = 0; j < outs[i].rows; ++j) {
            // Получаем данные для каждого объекта
            float confidence = data[j * 7 + 4];
            if (confidence > 0.5) { // Установка порога доверия
                int centerX = (int)(data[j * 7 + 0] * image.cols);
                int centerY = (int)(data[j * 7 + 1] * image.rows);
                int width = (int)(data[j * 7 + 2] * image.cols);
                int height = (int)(data[j * 7 + 3] * image.rows);
                int classId = (int)(data[j * 7 + 5]);

                // Вычисление координат объекта
                int x = centerX - width / 2;
                int y = centerY - height / 2;

                // Добавление объекта в список
                classIds.push_back(classId);
                confidences.push_back(confidence);
                boxes.push_back(Rect(x, y, width, height));
            }
        }
    }

    // Накладываем прямоугольники вокруг обнаруженных объектов
    for (size_t i = 0; i < boxes.size(); ++i) {
        rectangle(image, boxes[i], Scalar(0, 255, 0), 2);
    }

    // Вывод результата
    imshow("Распознавание объектов", image);
    waitKey(0);

    return 0;
}
