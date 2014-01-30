videoFilePath = fullfile('../output/mvi3177_blueWomanLane3.avi');
mmReader = mmreader(videoFilePath);
num = mmReader.NumberOfFrames;

vid1 = utils.VideoHelper.readFrames(videoFilePath, 1:29:29*16);

implay(immovie(vid1), 10)

% write output
writerObj = VideoWriter('../output/mvi3177_blueWomanLane3_ex1.avi');
writerObj.open();
writerObj.writeVideo(vid1);
writerObj.close();