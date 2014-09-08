/********************************************************************************
** Form generated from reading UI file 'TrackerArenaForm.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TRACKERARENAFORM_H
#define UI_TRACKERARENAFORM_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_TrackerArenaForm
{
public:

    void setupUi(QWidget *TrackerArenaForm)
    {
        if (TrackerArenaForm->objectName().isEmpty())
            TrackerArenaForm->setObjectName(QStringLiteral("TrackerArenaForm"));
        TrackerArenaForm->resize(400, 300);

        retranslateUi(TrackerArenaForm);

        QMetaObject::connectSlotsByName(TrackerArenaForm);
    } // setupUi

    void retranslateUi(QWidget *TrackerArenaForm)
    {
        TrackerArenaForm->setWindowTitle(QApplication::translate("TrackerArenaForm", "Form", 0));
    } // retranslateUi

};

namespace Ui {
    class TrackerArenaForm: public Ui_TrackerArenaForm {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TRACKERARENAFORM_H
