//Подключение заголовочных файлов и не конфликтующих пространств имен
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>
#include <ViZDoom.h>
#include <opencv2/opencv.hpp>
#include <PathTracer.h>

using namespace vizdoom;
using namespace cv;
using namespace std;

// Boilerplate методы и переменные связанные с настройкой VizDoom
shared_ptr<DoomGame> game; // Здесь будем хранить игру - это указатель
const int ButtonsCount = 42; // Введем 42 возможных действия (все доступные кнопки VizDoom)
vector<double> actions[ButtonsCount]; // Массив векторов в которых мы предсохраним возможные действия

// Действия будем сохранять в форматaке 0,0,...,0,1,0,...,0 где расположеие 1 будет отвечать за нажимаемую кнопку
void allowAllButtons() {
    vector<Button> n;
    game->setAvailableButtons(n);
    for(int i = 0; i < ButtonsCount; ++i ) {
        game->addAvailableButton((Button)i);
        actions[i] = vector<double>(42, 0);
        actions[i][i] = 1;
    }
}

//вызов действия - нажать на кнопку!
void PushButton(Button b) {
    game->makeAction(actions[b]);
}
//пропуск действия
void SkeepFrame() {
    static vector<double> noAction(42, 0);
    game->makeAction(noAction);
}

// Цифры которые в OpenCV соответствуют Клавишам
enum KEYS {
    SPACE = 32,
    LEFT = 97,
    RIGHT = 100,
    ROTATE_R = 101,
    ROTATE_L = 113,
    DOWN =  115 ,
    UP = 119
};


bool GetButton(const KEYS & action, Button & out) {
    bool result = true;
    switch(action) {
        case UP:  {out = MOVE_FORWARD; break;}
        case DOWN: { out = MOVE_BACKWARD;break;}
        case LEFT:  {out = MOVE_LEFT; break;}
        case RIGHT: {out = MOVE_RIGHT;break;}
        case SPACE:  {out = ATTACK;break;}
        case ROTATE_L:{out = TURN_LEFT;break;}
        case ROTATE_R:{out = TURN_RIGHT;break;}
        default: { result = false; break;}
    }
    return result;
}

int main() {
    game = make_shared<DoomGame>(); // Создаем инстанс игры

    try{ // Попробуем настроить параметры игры
        game->setViZDoomPath("/headless/base/ViZDoom/bin/vizdoom"); // путь к тому где лежит "сервер игры" который мы запустим локально
        game->setDoomGamePath("/headless/base/ViZDoom/bin/freedoom2.wad"); // путь к тому где все текстуры, базовые карты и логика игры
        game->loadConfig("/headless/base/ViZDoom/scenarios/2basic.cfg"); // путь к файлу настроек именно этой сессии
        allowAllButtons(); // Разрешим все кнопки (в не зависимости от того что было в конфигурации)
        game->setDepthBufferEnabled(true);
        game->setAutomapBufferEnabled(true);
        game->init(); // запустим платформу VizDoom
    } catch(exception &e) { // Поймаем ошибку если что-то пошло не так
        cout << e.what() << endl; // Выведем ее на экран
    }

    auto episodes = 10; // Система предпологает что игр будет проведено более одной
    auto sleepTime = 1000 / DEFAULT_TICRATE; // = 28 // Пока для простоты отладки мы заставляем сервер работать синхронно с нами
    // Другими словами пока мы не сказали - кадр кончился, новый не начнется. Эта переменная позволит нам создать аналог стабильного фреймрейта
    auto image = Mat(180, 320, CV_8UC3); // будем ранить здесь текущий кадр
    namedWindow( "Control Window", WINDOW_AUTOSIZE ); // Создадим OpenCV окно для приема управления от пользователя (клавиши WSAD)
    auto repeat_counter = 0; // Создадим переменную которая будет повторять последнее выполненное действие в течении некоторого времени
    for (auto i = 0; i < episodes; ++i) {
        cout << "Episode #" << i + 1 << "\n";
        game->newEpisode(); // Раунд пошел
        unique_ptr<PathTracer> path;
        vector<Mat> frames;

        while (!game->isEpisodeFinished()) { // Пока эпизод не кончился
            auto state = game->getState(); // Получим текущее состояние игрока\доступного нам мира

            //состоящего из:
            auto n              = state->number;
            auto vars       = state->gameVariables;
            auto screenBuf         = state->screenBuffer;
            auto depthBuf          = state->depthBuffer;
            auto labelsBuf         = state->labelsBuffer;
            auto automapBuf        = state->automapBuffer;
            // BufferPtr is shared_ptr<Buffer> where Buffer is vector<uint8_t>
            vector<Label> labels   = state->labels;

            image.data= static_cast<uchar*>(screenBuf->data()); // здесь мы подменяем указатель на данные
            auto im = image.clone(); // ViZDoom моет писать в тот буфер асинронно - сделаем локальную копию
            imshow("Control Window", im);

            Mat depth_src = Mat(180, 320, CV_8UC1), depth;
            depth_src.data= static_cast<uchar*>(depthBuf->data()); // здесь мы подменяем указатель на данные
                    depth = depth_src.clone();
            imshow("Depth Window", depth);

            Mat map_src = Mat(180, 320, CV_8UC3), map;
            map_src.data= static_cast<uchar*>(automapBuf->data()); // здесь мы подменяем указатель на данные
            map = map_src.clone();
            imshow("Map Window", map);

            auto key = waitKey(sleepTime)%256; // Ждем момента нажатия клавиши несколько миллисекунд
            if(key == 255 || key == -1 ) { // ненажали - продолжаем действие или спим
                if(repeat_counter-- > 0) {
                    game->advanceAction(1);
                } else {
                    SkeepFrame();
                }
            } else { // Нажали - смотрим что именно и либо выполняем либо ничего не делаем

                cout << "you pressed " << key << endl;
                auto action = static_cast<KEYS>(key);
                Button b;
                auto ok = GetButton(action, b);
                if(ok) {
                    PushButton(b);

                    // reconstruction
                    try {
                        if (frames.size() < 2) {
                            frames.push_back(im.clone());
                        } else if (frames.size() == 2) {
                            frames.push_back(im.clone());
                            path = make_unique<PathTracer>(frames[0], frames[1]);
                            path->addFrame(im.clone());
                        } else {
                            path->addFrame(im.clone());
                            path->RealAction(b);
                        }
                    }catch(std::exception &e) {
                        cout << e.what() << endl;
                    }
                    // manual reconstruction

                } else {
                    SkeepFrame();
                }




            }
        }

        cout << "Episode finished.\n";
        cout << "Total reward: " << game->getTotalReward() << "\n"; // правила выставления успешности конца задаются в конфигурации мира
    }
    game->close();
}
