/********************************************************************************
** Form generated from reading UI file 'TopViewDemoTracerMainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TOPVIEWDEMOTRACERMAINWINDOW_H
#define UI_TOPVIEWDEMOTRACERMAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "TrackerArenaForm.h"

QT_BEGIN_NAMESPACE

class Ui_TopViewDemoTracerMainWindow
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QWidget *widgetForButtons;
    QPushButton *pushButtonNextGameStep;
    QPushButton *pushButtonKillBug;
    QLabel *labelFrameInd;
    QLabel *label;
    QSpinBox *spinBoxBugNum;
    QPushButton *pushButtonPrintStat;
    QHBoxLayout *horizontalLayoutMain;
    QListWidget *listWidgetTracks;
    TrackerArenaForm *glWidgetTrace;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *TopViewDemoTracerMainWindow)
    {
        if (TopViewDemoTracerMainWindow->objectName().isEmpty())
            TopViewDemoTracerMainWindow->setObjectName(QStringLiteral("TopViewDemoTracerMainWindow"));
        TopViewDemoTracerMainWindow->resize(986, 592);
        centralWidget = new QWidget(TopViewDemoTracerMainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        widgetForButtons = new QWidget(centralWidget);
        widgetForButtons->setObjectName(QStringLiteral("widgetForButtons"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(widgetForButtons->sizePolicy().hasHeightForWidth());
        widgetForButtons->setSizePolicy(sizePolicy);
        widgetForButtons->setMinimumSize(QSize(0, 50));
        pushButtonNextGameStep = new QPushButton(widgetForButtons);
        pushButtonNextGameStep->setObjectName(QStringLiteral("pushButtonNextGameStep"));
        pushButtonNextGameStep->setGeometry(QRect(10, 10, 91, 23));
        pushButtonNextGameStep->setAutoRepeat(true);
        pushButtonNextGameStep->setDefault(true);
        pushButtonKillBug = new QPushButton(widgetForButtons);
        pushButtonKillBug->setObjectName(QStringLiteral("pushButtonKillBug"));
        pushButtonKillBug->setGeometry(QRect(120, 30, 75, 20));
        labelFrameInd = new QLabel(widgetForButtons);
        labelFrameInd->setObjectName(QStringLiteral("labelFrameInd"));
        labelFrameInd->setGeometry(QRect(500, 10, 151, 21));
        label = new QLabel(widgetForButtons);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(120, 10, 31, 16));
        spinBoxBugNum = new QSpinBox(widgetForButtons);
        spinBoxBugNum->setObjectName(QStringLiteral("spinBoxBugNum"));
        spinBoxBugNum->setGeometry(QRect(151, 9, 42, 22));
        pushButtonPrintStat = new QPushButton(widgetForButtons);
        pushButtonPrintStat->setObjectName(QStringLiteral("pushButtonPrintStat"));
        pushButtonPrintStat->setGeometry(QRect(414, 10, 81, 23));

        verticalLayout->addWidget(widgetForButtons);

        horizontalLayoutMain = new QHBoxLayout();
        horizontalLayoutMain->setSpacing(6);
        horizontalLayoutMain->setObjectName(QStringLiteral("horizontalLayoutMain"));
        listWidgetTracks = new QListWidget(centralWidget);
        listWidgetTracks->setObjectName(QStringLiteral("listWidgetTracks"));
        sizePolicy.setHeightForWidth(listWidgetTracks->sizePolicy().hasHeightForWidth());
        listWidgetTracks->setSizePolicy(sizePolicy);

        horizontalLayoutMain->addWidget(listWidgetTracks);

        glWidgetTrace = new TrackerArenaForm(centralWidget);
        glWidgetTrace->setObjectName(QStringLiteral("glWidgetTrace"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(glWidgetTrace->sizePolicy().hasHeightForWidth());
        glWidgetTrace->setSizePolicy(sizePolicy1);

        horizontalLayoutMain->addWidget(glWidgetTrace);


        verticalLayout->addLayout(horizontalLayoutMain);

        TopViewDemoTracerMainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(TopViewDemoTracerMainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 986, 21));
        TopViewDemoTracerMainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(TopViewDemoTracerMainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        TopViewDemoTracerMainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(TopViewDemoTracerMainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        TopViewDemoTracerMainWindow->setStatusBar(statusBar);

        retranslateUi(TopViewDemoTracerMainWindow);

        QMetaObject::connectSlotsByName(TopViewDemoTracerMainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *TopViewDemoTracerMainWindow)
    {
        TopViewDemoTracerMainWindow->setWindowTitle(QApplication::translate("TopViewDemoTracerMainWindow", "TopViewDemoTracerMainWindow", 0));
        pushButtonNextGameStep->setText(QApplication::translate("TopViewDemoTracerMainWindow", "NextGameStep", 0));
        pushButtonKillBug->setText(QApplication::translate("TopViewDemoTracerMainWindow", "Live On/Off", 0));
        labelFrameInd->setText(QApplication::translate("TopViewDemoTracerMainWindow", "TextLabel", 0));
        label->setText(QApplication::translate("TopViewDemoTracerMainWindow", "Bug #", 0));
        pushButtonPrintStat->setText(QApplication::translate("TopViewDemoTracerMainWindow", "Print Statistics", 0));
    } // retranslateUi

};

namespace Ui {
    class TopViewDemoTracerMainWindow: public Ui_TopViewDemoTracerMainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TOPVIEWDEMOTRACERMAINWINDOW_H
