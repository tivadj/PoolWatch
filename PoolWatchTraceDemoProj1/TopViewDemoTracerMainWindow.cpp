#include "topviewdemotracermainwindow.h"
#include "ui_topviewdemotracermainwindow.h"
#include <QDebug>
#include "TracingDemoModel.h"

TopViewDemoTracerMainWindow::TopViewDemoTracerMainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::TopViewDemoTracerMainWindow)
{
    ui->setupUi(this);
	QObject::connect(ui->pushButtonNextGameStep, SIGNAL(clicked()), this, SLOT(pushButtonNextGameStep_Pressed()));
	QObject::connect(ui->pushButtonKillBug, SIGNAL(clicked()), this, SLOT(pushButtonKillBug_Clicked()));
	QObject::connect(ui->pushButtonPrintStat, SIGNAL(clicked()), this, SLOT(pushButtonPrintStat_Clicked()));

	model_ = std::make_shared<TracingDemoModel>();

	auto interv = model_->interval();
	connect(&timer_, SIGNAL(timeout()), this, SLOT(timer_OnTimeout()));
	timer_.setInterval(interv);
	//timer_.start();

	QObject::connect(model_.get(), SIGNAL(changed()), this, SLOT(model_Changed()));
	QObject::connect(model_.get(), SIGNAL(trackAdded(int)), this, SLOT(model_trackAdded(int)));
	QObject::connect(model_.get(), SIGNAL(trackPruned(int)), this, SLOT(model_trackPruned(int)));

	ui->glWidgetTrace->setModel(model_.get());
}

TopViewDemoTracerMainWindow::~TopViewDemoTracerMainWindow()
{
    delete ui;
}

void TopViewDemoTracerMainWindow::pushButtonNextGameStep_Pressed()
{
	model_->nextGameStep(true);
}

void TopViewDemoTracerMainWindow::pushButtonKillBug_Clicked()
{
	int bugInd = ui->spinBoxBugNum->value();
	model_->switchBugLiveStatus(bugInd);
}

void TopViewDemoTracerMainWindow::pushButtonPrintStat_Clicked()
{
	model_->printStatistics();
}

void TopViewDemoTracerMainWindow::timer_OnTimeout()
{
	model_->nextGameStep(true);
}

void TopViewDemoTracerMainWindow::model_Changed()
{
	ui->glWidgetTrace->update();

    ui->labelFrameInd->setText(QString("FrameInd=%1 Ready=%2").arg(model_->frameInd_).arg(model_->readyFrameInd_));
}

void TopViewDemoTracerMainWindow::model_trackAdded(int trackId)
{
	TrackUIItem* pTrackObj = model_->getTrack(trackId);

	QListWidgetItem* item = new QListWidgetItem();
	item->setText(QString("Track %1").arg(trackId));
	item->setData(Qt::UserRole, QVariant(trackId));

	ui->listWidgetTracks->addItem(item);
}

void TopViewDemoTracerMainWindow::model_trackPruned(int trackId)
{
	TrackUIItem* pTrackObj = model_->getTrack(trackId);
	assert(!pTrackObj->IsLive);

	QListWidgetItem* trackItem = nullptr;
    for (int i = 0; i < ui->listWidgetTracks->count(); ++i)
    {
		QListWidgetItem* item = ui->listWidgetTracks->item(i);
		if (item->data(Qt::UserRole) == QVariant(trackId))
		{
			trackItem = item;
			break;
		}
    }
	if (trackItem != nullptr)
	{
		trackItem->setText(QString("Track %1 zombie").arg(trackId));
	}
}

void TopViewDemoTracerMainWindow::mainWindow_Resized()
{
}

void TopViewDemoTracerMainWindow::resizeEvent(QResizeEvent*)
{
}
