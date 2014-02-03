% Retrieves images and masks filtered by strokeColor.
function [image,imageMask] = getMaskAll(xmlSvgFilePath,strokeColor)

if ~exist('strokeColor')
    strokeColor = '';
end

% construct DocumentBuilder which will not access external DTDs
dbf = javax.xml.parsers.DocumentBuilderFactory.newInstance;
dbf.setFeature('http://apache.org/xml/features/nonvalidating/load-external-dtd', false);

db = dbf.newDocumentBuilder;

% load xml
xDoc = xmlread(xmlSvgFilePath, db);
xDocText = xDoc.getTextContent;
ims=xDoc.getElementsByTagName('image');

% extract image
if ims.getLength ~= 1
    error('file %s contains multiple images', xmlSvgFilePath);
end    

imageFilePath = ims.item(0).getAttribute('xlink:href');
imageFilePath = char(imageFilePath);
imageFilePath = fullfile(fileparts(xmlSvgFilePath), imageFilePath);
imageInfo = imfinfo(imageFilePath);

% load image
image = imread(imageFilePath);

height = imageInfo.Height;
width = imageInfo.Width;
imageMask = zeros(height, width, 'uint8');

% retrieve polygons
% process polyline too?
pols=xDoc.getElementsByTagName('polygon');
if pols.getLength == 0
    %error('There are no polygons in given SVG file %s', xmlSvgFilePath);
end

for k=1:pols.getLength
    node = pols.item(k-1);

    % filter curves by queried color
    strColStr = char(node.getAttribute('stroke'));
    if ~isempty(strokeColor) && ~strcmpi(strColStr, strokeColor)
        continue;
    end

    pointsStr = node.getAttribute('points');
    pointsStr = char(pointsStr);
    pointsStrArray = strsplit(pointsStr, {' ',','}, 'CollapseDelimiters',true);
    points = str2double(pointsStrArray);
    points = points(~isnan(points)); % remove NaN
    points = reshape(points, 2, [])';

    % poly
    mask1 = poly2mask(points(:,1), points(:,2), height, width);
    imageMask = imageMask | mask1;
end

end
