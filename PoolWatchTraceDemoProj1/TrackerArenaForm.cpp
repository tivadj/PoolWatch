#include "TrackerArenaForm.h"
#include "ui_TrackerArenaForm.h"

#include <cmath>
#include <array>
#include <gl/GLU.h>
#include <QPainter>
#include <QDebug>
#include "TracingDemoModel.h"

TrackerArenaForm::TrackerArenaForm(QWidget *parent) :
    QGLWidget(parent),
    ui(new Ui::TrackerArenaForm)
{
    ui->setupUi(this);

	setAutoFillBackground(false);
}

TrackerArenaForm::~TrackerArenaForm()
{
    delete ui;
}

void TrackerArenaForm::initializeGL()
{
    glClearColor(1,1,1,1);
    glPointSize(5);
}

void TrackerArenaForm::resizeGL(int width, int height)
{
	qDebug() << "TrackerArenaForm::resizeGL (" << width << ", " << height << ")";
	setupViewport(width, height);
}

void TrackerArenaForm::setupViewport(int width1, int height1)
{
	//qDebug() << "TrackerArenaForm::setupViewport (" << width1 << ", " << height1 << ")";
	glViewport(0, 0, width1, height1);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, width1, 0, height1);
	
	// put origin to top-left of the widget
	glTranslatef(0, height1, 0);
	glScalef(1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void TrackerArenaForm::paintBugs(const std::vector<BugCreature>& bugs, uchar alpha)
{
	glBegin(GL_POINTS);
	int i = 0;
	for (const auto& bug : bugs)
	{
		if (!bug.Visible)
			continue;

		glColor4ub(bug.PrimaryColor[0],bug.PrimaryColor[1],bug.PrimaryColor[2],alpha);
		glVertex2f(bug.Pos.x(), bug.Pos.y());

		++i;
	}
	glEnd();
}

void TrackerArenaForm::paintBlobs(const std::vector<DetectedBlob>& blobs, uchar alpha)
{
	for (const DetectedBlob& blob : blobs)
	{
		cv::Vec3b blobFirstPix3 = blob.FilledImageRgb.at<cv::Vec3b>(0, 0);

		std::array<uchar, 4> pixColor4 = { blobFirstPix3[0], blobFirstPix3[1], blobFirstPix3[2] ,alpha};
		glColor4ubv((GLubyte*)pixColor4.data());

		// draw detected blob as a box
		glBegin(GL_LINE_LOOP);
		glVertex2f(blob.BoundingBox.x, blob.BoundingBox.y);
		glVertex2f(blob.BoundingBox.x + blob.BoundingBox.width, blob.BoundingBox.y);
		glVertex2f(blob.BoundingBox.x + blob.BoundingBox.width, blob.BoundingBox.y + blob.BoundingBox.height);
		glVertex2f(blob.BoundingBox.x, blob.BoundingBox.y + blob.BoundingBox.height);
		glEnd();
	}
	
}

void TrackerArenaForm::paintGLInternal()
{
	glClear(GL_COLOR_BUFFER_BIT);

	glBegin(GL_LINES);
	glColor3f(1, 0, 0);
	glVertex2f(5, 5);
	glVertex2f(315, 195);
	glEnd();

	// draw what is actually visible now
	if (model_->showCurrentBlobs_)
	{
		const auto& curBugs = model_->bugsHistory_->queryHistory(0);
		paintBugs(curBugs, 55);

		const auto& curBlobs = model_->blobsHistory_->queryHistory(-0);
		paintBlobs(curBlobs, 55);
	}

	const auto& historyBugs = model_->bugsHistory_->queryHistory(-model_->pruneWindow_);
	paintBugs(historyBugs, 255);

	const auto& historyBlobs = model_->blobsHistory_->queryHistory(-model_->pruneWindow_);
	paintBlobs(historyBlobs, 255);
	//

	// draw what was visible model.pruneWindow_ frames ago
	if (model_->readyFrameInd_ >= 0)
	{
		//const auto& readyBugs = model_->bugsPerFrame_[model_->readyFrameInd_];
		//paintBugs(readyBugs, 256);
	}
}

void TrackerArenaForm::overlapOpenGL(QPainter& painter)
{
	if (model_->readyFrameInd_ < 0)
		return;
	
	// draw track adornments
	const float r = 5;

	std::vector<TrackUIItem*> trackList;
	model_->getTrackList(trackList);
	for (auto pTrack : trackList)
	{
		BlobRegistrationRecord blobInfo = pTrack->LastPosition;
		auto pos = blobInfo.ObsPosPixExactOrApprox;

		if (!pTrack->IsLive)
		{
			painter.setPen(QColor::fromRgb(0, 0, 0));
			painter.drawEllipse(pos.x - r, pos.y - r, 2 * r, 2 * r);
		}
		else
		{
			painter.setPen(QColor::fromRgb(0, 0, 0));
			painter.drawText(pos.x, pos.y, QString::number(pTrack->TrackId));

			QColor penColor;
			if (blobInfo.HasObservation)
				penColor = QColor::fromRgba(qRgba(0, 255, 0, 255));
			else
				penColor = QColor::fromRgb(qRgba(0, 0, 255, 255));
			
			painter.setPen(penColor);

			painter.drawEllipse(pos.x - r, pos.y - r, 2 * r, 2 * r);
		}

		// attempt to estimate most probable hypothesis for current track
		//int latestFrameInd = pTrack->latesetFrameIndExcl();

		//TrackChangePerFrame change;
		//if (model_->getLatestTrack(latestFrameInd, pTrack->TrackId, change))
		//{
		//	auto pos = change.ObservationPosPixExactOrApprox;

		//	painter.setPen(QColor::fromRgb(0, 0, 0));
		//	painter.drawText(pos.x, pos.y, QString::number(pTrack->TrackId));

		//	QColor penColor;
		//	if (change.UpdateType == TrackChangeUpdateType::ObservationUpdate || change.UpdateType == TrackChangeUpdateType::New)
		//		penColor = QColor::fromRgb(0, 255, 0);
		//	else if (change.UpdateType == TrackChangeUpdateType::NoObservation)
		//		penColor = QColor::fromRgb(255, 255, 0);
		//	else if (change.UpdateType == TrackChangeUpdateType::Pruned)
		//		penColor = QColor::fromRgb(255, 0, 0);
		//	painter.setPen(penColor);
		//	
		//	painter.drawEllipse(pos.x - r, pos.y - r, 2 * r, 2 * r);
		//}
		//else
		//{
		//	qDebug() << "Can't get latest track hypothesis TrackId=" << pTrack->TrackId;
		//	CV_Assert(false);
		//}
	}
}

void TrackerArenaForm::paintGL()
{
}

void TrackerArenaForm::paintEvent(QPaintEvent* e)
{
	//qDebug() << "paintEvent1";

	makeCurrent();

//	glMatrixMode(GL_MODELVIEW);
//	glPushMatrix();
//	glMatrixMode(GL_PROJECTION);
//	glPushMatrix();

	initializeGL();

	float ww = width();
	float hh = height();
	setupViewport(ww, hh);
	paintGLInternal();

//	glMatrixMode(GL_MODELVIEW);
//	glPopMatrix();
//	glMatrixMode(GL_PROJECTION);
//	glPopMatrix();

	//
	QPainter painter(this);
	overlapOpenGL(painter);
	//painter.end();

	//qDebug() << "paintEvent2";
}