classdef BlobsHelperTest < matlab.unittest.TestCase

methods(Test)

function RussianDoll(testCase)
    [~,mask1]=utils.getMaskAll('data/tests/MergeBlobs/MergeBlobs_RussianDoll.svg', '#FFFFFF');
    connComps = bwconncomp(mask1,4);
    
    mergedBlobsMask = utils.BlobsHelper.mergeBlobs(connComps, 1, 2);
    testCase.verifyEqual(mergedBlobsMask(100,150), true);
end

function TwoPrisms(testCase)
    [~,mask1]=utils.getMaskAll('data/tests/MergeBlobs/MergeBlobs_TwoPrisms.svg', '#FFFFFF');
    connComps = bwconncomp(mask1,4);
    
    mergedBlobsMask = utils.BlobsHelper.mergeBlobs(connComps, 1, 2);
    testCase.verifyEqual(mergedBlobsMask(47,185), true);
end

function bodyParts(testCase)
    debug = true;
    imageRgb = imread('data/tests/MergeBlobs/bodyParts1_simple1.png');
    imageGray = rgb2gray(imageRgb);
    connComp = bwconncomp(imageGray, 8);
    
    hd = HumanDetector([], [], []);
    
    maxDistBetweenBlobs = 9999;
    maxBlobColorMergeDist = 15;
    mergeRecipe = hd.mergeBlobsRecipeViaColorSimilarity(connComp, [], imageRgb, maxDistBetweenBlobs, maxBlobColorMergeDist, debug);
    testCase.verifyEqual(size(mergeRecipe,1), 3);
end

end
    
end
