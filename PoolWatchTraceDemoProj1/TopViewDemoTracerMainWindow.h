#ifndef TOPVIEWDEMOTRACERMAINWINDOW_H
#define TOPVIEWDEMOTRACERMAINWINDOW_H

#include <memory>
#include <QMainWindow>
#include <QtWidgets/QMainWindow> // TODO: which QMainWindow
#include <QTimer>

#include "TracingDemoModel.h"

namespace Ui {
class TopViewDemoTracerMainWindow;
}

class TopViewDemoTracerMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit TopViewDemoTracerMainWindow(QWidget *parent = 0);
    ~TopViewDemoTracerMainWindow();

private slots:
	void pushButtonNextGameStep_Pressed();
	void pushButtonKillBug_Clicked();
	void pushButtonPrintStat_Clicked();
	void timer_OnTimeout();
	void model_Changed();
	void model_trackAdded(int trackId);
	void model_trackPruned(int trackId);
	void mainWindow_Resized();

protected:
	void resizeEvent(QResizeEvent*) override;

private:
    Ui::TopViewDemoTracerMainWindow *ui;
	QTimer timer_;
	std::shared_ptr<TracingDemoModel> model_;
};

#endif // TOPVIEWDEMOTRACERMAINWINDOW_H
