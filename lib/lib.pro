#-------------------------------------------------
#
# Project created by QtCreator 2018-05-10T11:53:56
#
#-------------------------------------------------

QT       -= core gui

TARGET = fasterRcnn
TEMPLATE = lib

DEFINES += FASTERRCNN_LIBRARY

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += fasterrcnn.cpp

HEADERS += fasterrcnn.h\
        fasterrcnn_global.h

unix {
    CONFIG(debug, debug|release) {
        DESTDIR = $$PWD/../build/debug
    } else {
        DESTDIR = $$PWD/../build/release
    }

}



LIBS += -lcaffe
LIBS += -lcudart -lcublas -lcudnn -lcurand



ubuntu_lirui{

#caffe
INCLUDEPATH += /home/lirui/packages/caffe_mtcnn/include
LIBS += -L/home/lirui/packages/caffe_mtcnn/build/lib


#cuda
INCLUDEPATH += /usr/local/cuda/include
LIBS += -L/usr/local/cuda/lib64
}
