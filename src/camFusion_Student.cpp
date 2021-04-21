#include <string>
#include <set>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

#include <opencv2/features2d.hpp>

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, string imgNumber)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
        putText(topviewImg, imgNumber, cv::Point2f(left - 250, bottom - 25), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    if (bWait)
    {
        // display image
        string windowName = "3D Objects";
        cv::namedWindow(windowName, 1);
        cv::imshow(windowName, topviewImg);
        string filename = "./images/saved/3dObjects_" + imgNumber + ".jpg";
        //cv::imwrite(filename, topviewImg);
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    bool verbose = false;
    // compute a robust mean of all the euclidean distances between keypoint matches
    // and then remove those that are too far away from the mean.

    int num_points = 0;
    float tot_dist = 0.0;

    for (cv::DMatch match : kptMatches)
    {
        cv::KeyPoint keypoint = kptsCurr[match.trainIdx];

        //checking whether the corresponding keypoints are within the region of interest in the camera image
        if (boundingBox.roi.contains(keypoint.pt))
        {

            cv::KeyPoint prev_keypoint = kptsPrev[match.queryIdx];

            tot_dist += cv::norm(keypoint.pt - prev_keypoint.pt);
            num_points += 1;
        }
    }

    float mean_dist = tot_dist / (float)num_points;

    if (verbose)
    {
        cout << "pts: " << num_points << "\tavg dist: " << mean_dist << endl;
    }

    for (cv::DMatch match : kptMatches)
    {
        cv::KeyPoint keypoint = kptsCurr[match.trainIdx];
        cv::KeyPoint prev_keypoint = kptsPrev[match.queryIdx];

        float dist = cv::norm(keypoint.pt - prev_keypoint.pt);

        //checking whether the corresponding keypoints are within the region of interest in the camera image
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt) && dist <= mean_dist)
        {
            //All matches which satisfy this condition should be added to a vector
            boundingBox.kptMatches.push_back(match);
            boundingBox.keypoints.push_back(keypoint);
        }
    }

    if (verbose)
        cout << boundingBox.keypoints.size() << endl;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{

    // compute the lowest mean of all euclidean distances between keypoint matches and
    // remove those that are too far away from the mean

    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}

bool sortfunction(LidarPoint i, LidarPoint j) { return (i.x < j.x); }

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, double &TTC_min)
{
    // auxiliary variables
    double dT = 0.1;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        minXPrev = minXPrev > it->x ? it->x : minXPrev;
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        minXCurr = minXCurr > it->x ? it->x : minXCurr;
    }
    // compute TTC from both measurements
    TTC_min = minXCurr * dT / (minXPrev - minXCurr);

    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), sortfunction);
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), sortfunction);

    double median_prev = lidarPointsPrev[lidarPointsPrev.size() / 2].x;
    double median_curr = lidarPointsCurr[lidarPointsCurr.size() / 2].x;

    // compute TTC from both measurements
    TTC = median_curr * dT / (median_prev - median_curr);
}

void DataFrameDetails(DataFrame &df)
{
    std::cout << "keypoints: \t" << df.keypoints.size() << std::endl;
    std::cout << "kptMatches: \t" << df.kptMatches.size() << std::endl;
    std::cout << "lidarPoints: \t" << df.lidarPoints.size() << std::endl;
    std::cout << "boundingBoxes: \t" << df.boundingBoxes.size() << std::endl;
    std::cout << std::endl;
}

void BoundingBoxDetails(BoundingBox &bb)
{
    std::cout << "boxID: \t" << bb.boxID << std::endl;
    std::cout << "trackID: \t" << bb.trackID << std::endl;
    std::cout << "roi: \t" << bb.roi << std::endl;
    std::cout << "classID: \t" << bb.classID << std::endl;
    std::cout << "confidence: \t" << bb.confidence << std::endl;
    std::cout << "lidarPoints: \t" << bb.lidarPoints.size() << std::endl;
    std::cout << "keypoints: \t" << bb.keypoints.size() << std::endl;
    std::cout << "kptMatches: \t" << bb.kptMatches.size() << std::endl;
    std::cout << std::endl;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, string imgNumber, string detector, string descriptor)
{
    // basic Idea = use keypoint matches between the previous and current image
    std::multimap<int, int> potential_matches;
    typedef std::multimap<int, int>::iterator MMAPIterator;

    bool verbose = true;

    if (verbose)
    {
        std::cout << "currFrame: " << std::endl;
        DataFrameDetails(currFrame);
        std::cout << "prevFrame: " << std::endl;
        DataFrameDetails(prevFrame);
    }

    if (verbose)
    {
        for (const auto &bb : currFrame.boundingBoxes)
        {
            std::cout << "curr boundingBoxes: id: ";
            std::cout << bb.boxID << '\t' << bb.roi << std::endl;
        }
        for (const auto &kp : currFrame.keypoints)
        {
            std::cout << "keypoint: ";
            std::cout << "x: " << kp.pt.x << "\ty: " << kp.pt.y << std::endl;
        }

        for (const auto &bb : prevFrame.boundingBoxes)
        {
            std::cout << "prev boundingBoxes: id: ";
            std::cout << bb.boxID << '\t' << bb.roi << std::endl;
        }
        for (const auto &kp : prevFrame.keypoints)
        {
            std::cout << "keypoint: ";
            std::cout << "x: " << kp.pt.x << "\ty: " << kp.pt.y << std::endl;
        }
    }

    //outer loop over keypoint matches
    if (verbose)
        std::cout << "matches: " << std::endl;

    for (const auto &it : matches)
    {
        int id_prev = -1;
        int id_curr = -1;

        //try to find out by which bounding boxes keypoints are enclosed both on the previous and current frame (Potential match cantidates)

        // https://docs.opencv.org/3.4/d2/d44/classcv_1_1Rect__.html#details
        //  OpenCV typically assumes that the top and left boundary of the rectangle are inclusive, while the right and bottom boundaries are not. For example, the method Rect_::contains returns true if
        //  x≤pt.x<x+width,y≤pt.y<y+height
        cv::KeyPoint kp_prev = prevFrame.keypoints[it.queryIdx];
        for (auto &bb_prev : prevFrame.boundingBoxes)
        {

            if (bb_prev.roi.contains(kp_prev.pt))
            {
                id_prev = bb_prev.boxID;
            }
        }

        cv::KeyPoint kp_curr = currFrame.keypoints[it.trainIdx];
        for (auto &bb_curr : currFrame.boundingBoxes)
        {
            if (bb_curr.roi.contains(kp_curr.pt))
            {
                id_curr = bb_curr.boxID;
            }
        }

        //store potential matches in a multimap
        if (id_prev != -1)
        {
            potential_matches.insert(std::make_pair(id_curr, id_prev));
        }
    }

    if (verbose)
    {
        for (auto &pm : potential_matches)
        {
            std::cout << pm.first << "," << pm.second << "\t";
        }
        std::cout << endl;
    }

    vector<int> box_ids;

    if (verbose)
        std::cout << "BoxID: ";

    for (auto &bb : currFrame.boundingBoxes)
    {
        box_ids.push_back(bb.boxID);
        if (verbose)
            std::cout << "\t" << bb.boxID;

        bool bVis = false;
        // visualize results
        if (bVis)
        {
            cv::Mat visImage = currFrame.cameraImg.clone();
            cv::drawKeypoints(visImage, bb.keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::rectangle(visImage, bb.roi, cv::Scalar(0, 0, 255), 2);
            cv::putText(visImage, to_string(bb.boxID), cv::Point(bb.roi.x, bb.roi.y), cv::FONT_ITALIC, 1.0, cv::Scalar(0, 0, 255), 2);

            string windowName = "BB Keypoints";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            string filename = "./images/saved/BB/BB_" + imgNumber + "_" + to_string(bb.boxID) + ".jpg";

            cv::imwrite(filename, visImage);
            //std::cout << filename << endl;
            //cv::waitKey(0);
        }
    }
    std::cout << endl;

    //find all matches in the multimap which share the same bounding box ID in the previous frame and count them

    for (const auto &it_boxid : box_ids)
    {
        multimap<int, int> map_matches;

        auto prev_ids = potential_matches.equal_range(it_boxid);

        //http://www.cplusplus.com/reference/map/multimap/equal_range/
        std::pair<std::multimap<int, int>::iterator, std::multimap<int, int>::iterator> ret;
        ret = potential_matches.equal_range(it_boxid);
        vector<int> prev_box_id, count_previd;
        if (verbose)
            std::cout << it_boxid << " =>";
        for (std::multimap<int, int>::iterator it = ret.first; it != ret.second; ++it)
        {
            prev_box_id.push_back(it->second);
            if (verbose)
                std::cout << ' ' << it->second;
        }

        if (verbose)
            std::cout << '\n';

        std::map<int, int> frequency;
        for (int i : prev_box_id)
            ++frequency[i];

        if (verbose)
        {
            for (const auto &e : frequency)
                std::cout << e.first << " - " << e.second << "\n";
        }
        //associate the bounding boxes with the highest number of occurances
        using pair_type = decltype(frequency)::value_type;
        auto pr = std::max_element(
            std::begin(frequency), std::end(frequency),
            [](const pair_type &p1, const pair_type &p2) {
                return p1.second < p2.second;
            });

        if (verbose)
            std::cout << "Mode: " << pr->first << '\n';

        //return box ids of all matched pairs im map bbBestMatches
        bbBestMatches.insert(std::pair<int, int>(pr->first, it_boxid));

        // Print the vector
        if (verbose)
        {
            cout << endl
                 << "BoxID: ";
            for (int i = 0; i < count_previd.size(); i++)
                cout << i << "=>" << count_previd[i] << endl;

            std::cout << '\n';
        }
    }
    std::cout << std::endl;

    if (verbose)
    {
        cout << "\nBoxID: ";
        std::cout << "prev\tcurr" << std::endl;
        for (const auto &p : bbBestMatches)
        {
            std::cout << p.first << '\t' << p.second << std::endl;
        }
    }
}
