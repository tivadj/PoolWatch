#ifndef TRACKERARENAFORM_H
#define TRACKERARENAFORM_H

#include <QGLWidget>
#include "TracingDemoModel.h"

namespace Ui {
class TrackerArenaForm;
}

class TrackerArenaForm : public QGLWidget
{
    Q_OBJECT

public:
    explicit TrackerArenaForm(QWidget *parent = 0);
    ~TrackerArenaForm();

	void setModel(TracingDemoModel* model) { model_ = model; }

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
	void setupViewport(int width, int height);
	void paintBugs(std::vector<BugCreature> const& bugs, uchar alpha);
	void paintBlobs(const std::vector<DetectedBlob>& blobs, uchar alpha);
	void paintGLInternal();
	void overlapOpenGL(QPainter& painter);
	void paintGL() override;

	void paintEvent(QPaintEvent*) override;
private:
	
private:
    Ui::TrackerArenaForm *ui;
	TracingDemoModel* model_;
};

#endif // TRACKERARENAFORM_H
