#include <random>
#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2\opencv.hpp>


using namespace cv;


double Mean(Mat& image)
{
    int channels = image.channels();
    long mean = 0;
    for (auto i = 0; i < image.rows; i++)
    {
        for (auto j = 0; j < image.cols; j++)
        {
            Vec3b& color = image.at<Vec3b>(i, j);
            for (auto ch = 0; ch < channels; ch++)
            {
                mean += color[ch];
            }
        }
    }
    return (mean / (image.rows * image.cols * channels));
}


double Sigma(Mat& image, int mean)
{
    int channels = image.channels();
    long sigma = 0;
    for (auto i = 0; i < image.rows; i++)
    {
        for (auto j = 0; j < image.cols; j++)
        {
            Vec3b& color = image.at<Vec3b>(i, j);
            for (auto ch = 0; ch < channels; ch++)
            {
                sigma += (color[ch] - mean) * (color[ch] - mean);
            }
        }
    }
    sigma /= (image.rows * image.cols * channels);
    return sqrt(sigma);
}


void Monochrome(Mat& image)
{
    Mat mono(image.rows, image.cols, 0.0);
    for (auto i = 0; i < image.rows; i++)
    {
        for (auto j = 0; j < image.cols; j++)
        {
            mono.at<uchar>(i, j) = (image.at<Vec3b>(i, j)[0] + image.at<Vec3b>(i, j)[1] + image.at<Vec3b>(i, j)[2]) / 3;
        }
    }
    mono.copyTo(image);
}


void SaltPepper(Mat& image, float noise)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib1(0, image.rows - 1);
    std::uniform_int_distribution<> distrib2(0, image.cols - 1);
    std::uniform_int_distribution<> distrib3(0, 1);

    auto amount = static_cast<int>(noise * image.rows * image.cols);
    for (auto i = 0; i < amount; i++)
    {
        auto row = distrib1(gen);

        auto column = distrib2(gen);

        auto x = distrib3(gen);

        auto y = (x ? 255 : 0);

        Vec3b& color = image.at<Vec3b>(row, column);

        color[0] = y;
        color[1] = y;
        color[2] = y;
    }
}


void GaussianNoise(Mat& image)
{
    auto mean = Mean(image);
    auto sigma = Sigma(image, mean);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distrib(mean, sigma);

    auto imageChannels = image.channels();

    for (auto row = 0; row < image.rows; row++)
    {
        for (auto column = 0; column < image.cols; column++)
        {
            Vec3b& color = image.at<Vec3b>(row, column);
            for (auto channel = 0; channel < imageChannels; channel++)
            {
                int newColor = color[channel] + distrib(gen);
                if (newColor > 255)
                {
                    color[channel] = 255;
                }
                else if (newColor < 0)
                {
                    color[channel] = 0;
                }
                else
                {
                    color[channel] = newColor;
                }
            }
        }
    }
}


void MedianFilter(Mat& image, int halfSize)
{
    Mat tempImage;

    image.copyTo(tempImage);

    auto imageChannels = image.channels();

    int kernelSize = halfSize * 2 + 1;

    auto totalKernelElements = kernelSize * kernelSize;

    auto pos = totalKernelElements / 2;

    for (auto i = halfSize; i < tempImage.rows - halfSize; i++)
    {
        for (auto j = halfSize; j < tempImage.cols - halfSize; j++)
        {
            std::vector<std::vector<int>> values(imageChannels, std::vector<int>(totalKernelElements));

            auto index = 0;

            for (auto x = -halfSize; x <= halfSize; x++)
            {
                for (auto y = -halfSize; y <= halfSize; y++)
                {
                    for (int channel = 0; channel < imageChannels; channel++)
                    {
                        unsigned char* pixelValuePtr = tempImage.ptr(i + x) + ((j + y) * imageChannels) + channel;

                        values[channel][index] = *pixelValuePtr;
                    }
                    index++;
                }
            }

            for (auto channel = 0; channel < imageChannels; channel++)
            {
                sort(begin(values[channel]), end(values[channel]));

                unsigned char* pixelValuePtr = image.ptr(i) + (j * imageChannels) + channel;

                *pixelValuePtr = values[channel][pos];
            }
        }
    }
}


void MeanFilter(Mat& image, int kernelSize)
{
    Mat tempImage;

    image.copyTo(tempImage);

    auto totalKernelElements = kernelSize * kernelSize;

    std::vector<double> kernel(totalKernelElements, 1.0 / totalKernelElements);

    auto imageChannels = image.channels();

    std::vector<std::vector<int>> values(imageChannels);

    int halfSize = kernelSize / 2;

    for (auto i = halfSize; i < tempImage.rows - halfSize; i++)
    {
        for (auto j = halfSize; j < tempImage.cols - halfSize; j++)
        {

            for (auto channel = 0; channel < imageChannels; channel++)
            {
                values[channel].clear();
            }

            for (auto x = -halfSize; x <= halfSize; x++)
            {
                for (auto y = -halfSize; y <= halfSize; y++)
                {
                    for (auto channel = 0; channel < imageChannels; channel++)
                    {
                        unsigned char* pixelValuePtr = tempImage.ptr(i + x) + ((j + y) * imageChannels) + channel;

                        values[channel].push_back(*pixelValuePtr);
                    }
                }
            }

            for (auto channel = 0; channel < imageChannels; channel++)
            {
                std::vector<int> channelValues = values[channel];

                long newPixelValue = std::inner_product(begin(channelValues), end(channelValues), begin(kernel), 0);

                unsigned char* pixelValuePtr = image.ptr(i) + (j * imageChannels) + channel;

                if (newPixelValue > 255)
                {
                    *pixelValuePtr = 255;
                }
                else if (newPixelValue < 0)
                {
                    *pixelValuePtr = 0;
                }
                else
                {
                    *pixelValuePtr = newPixelValue;
                }
            }
        }
    }
}


void GaussianFilter(Mat& image, int kernelSize)
{
    Mat tempImage;

    image.copyTo(tempImage);

    auto totalKernelElements = kernelSize * kernelSize;

    std::vector<double> kernel(totalKernelElements, 1.0 / totalKernelElements);

    auto imageChannels = image.channels();

    std::vector<std::vector<int>> values(imageChannels);

    int halfSize = kernelSize / 2;

    for (auto i = halfSize; i < tempImage.rows - halfSize; i++)
    {
        for (auto j = halfSize; j < tempImage.cols - halfSize; j++)
        {
            for (auto channel = 0; channel < imageChannels; channel++)
            {
                values[channel].clear();
            }

            for (auto x = -halfSize; x <= halfSize; x++)
            {
                for (auto y = -halfSize; y <= halfSize; y++)
                {
                    for (auto channel = 0; channel < imageChannels; channel++)
                    {
                        unsigned char* pixelValuePtr = tempImage.ptr(i + x) + ((j + y) * imageChannels) + channel;

                        values[channel].push_back(*pixelValuePtr);
                    }
                }
            }

            for (auto channel = 0; channel < imageChannels; channel++)
            {
                std::vector<int> channelValues = values[channel];

                long mean = std::inner_product(begin(channelValues), end(channelValues), begin(kernel), 0);

                for (auto k = 0; k != channelValues.size(); k++)
                {
                    channelValues[k] = (channelValues[k] - mean) * (channelValues[k] - mean);
                }

                long sigma = sqrt(std::inner_product(begin(channelValues), end(channelValues), begin(kernel), 0));

                unsigned char* pixelValuePtr = image.ptr(i) + (j * imageChannels) + channel;

                long newPixelValue = *pixelValuePtr + sigma;

                if (newPixelValue > 255)
                {
                    *pixelValuePtr = 255;
                }
                else if (newPixelValue < 0)
                {
                    *pixelValuePtr = 0;
                }
                else
                {
                    *pixelValuePtr = newPixelValue;
                }
            }
        }
    }
}


void Detection(Mat& image, std::vector<int>& xKernel, std::vector<int>& yKernel)
{
    Mat tempImage;

    if (image.channels() != 1)
    {
        Monochrome(image);
    }

    image.copyTo(tempImage);

    int kernelSize = sqrt(xKernel.size());

    std::vector<int> values;

    int halfSize = kernelSize / 2;

    for (int i = halfSize; i < tempImage.rows - halfSize; i++)
    {
        for (int j = halfSize; j < tempImage.cols - halfSize; j++)
        {
            values.clear();

            for (int x = -halfSize; x <= halfSize; x++)
            {
                for (int y = -halfSize; y <= halfSize; y++)
                {
                    unsigned char* pixelValuePtr = tempImage.ptr(i + x) + (j + y);

                    values.push_back(*pixelValuePtr);
                }
            }

            long gX = inner_product(begin(values), end(values), begin(xKernel), 0);

            long gY = inner_product(begin(values), end(values), begin(yKernel), 0);

            long newPixelValue = abs(gX) + abs(gY);

            unsigned char* pixelValuePtr = image.ptr(i) + j;

            if (newPixelValue > 255)
            {
                *pixelValuePtr = 255;
            }
            else if (newPixelValue < 0)
            {
                *pixelValuePtr = 0;
            }
            else
            {
                *pixelValuePtr = newPixelValue;
            }
        }
    }
}


void Laplacian(Mat& image)
{
    std::vector<int> xKernel = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };

    std::vector<int> yKernel = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

    Detection(image, xKernel, yKernel);

    image = image > 30;
}


void Sobel(Mat& image)
{
    std::vector<int> xKernel = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };

    std::vector<int> yKernel = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

    Detection(image, xKernel, yKernel);

    image = image > 100;
}


void HistogramEqualization(Mat& image)
{
    Mat tempImage;

    image.copyTo(tempImage);

    auto imageChannels = image.channels();

    std::vector<std::vector<int>> values(imageChannels, std::vector<int>(tempImage.rows * tempImage.cols));
    auto index = 0;
    for (auto i = 0; i < tempImage.rows; i++)
    {
        for (auto j = 0; j < tempImage.cols; j++)
        {
            for (auto channel = 0; channel < imageChannels; channel++)
            {
                unsigned char* pixelValuePtr = tempImage.ptr(i) + (j * imageChannels) + channel;

                values[channel][index] = *pixelValuePtr;
            }
            index++;
        }
    }
    std::vector<std::vector<double>> amount(imageChannels, std::vector<double>(255, 0.0));
    for (auto i = 0; i < 255; i++)
    {
        for (auto channels = 0; channels < imageChannels; channels++)
        {
            amount[channels][i] = std::count(begin(values[channels]), end(values[channels]), i);
            amount[channels][i] /= tempImage.rows * tempImage.cols;
            amount[channels][i] = std::round(amount[channels][i] * 10000) / 10000;
            if (amount[channels][i] > 1.0)
            {
                amount[channels][i] = 0.0;
            }
            if (i > 1)
            {
                amount[channels][i] += amount[channels][i - 1];
            }
        }
    }
    for (auto i = 0; i < 255; i++)
    {
        for (auto channels = 0; channels < imageChannels; channels++)
        {
            amount[channels][i] *= 255;
            amount[channels][i] = std::round(amount[channels][i]);
        }
    }
    index = 0;
    for (auto i = 0; i < tempImage.rows; i++)
    {
        for (auto j = 0; j < tempImage.cols; j++)
        {
            for (auto channel = 0; channel < imageChannels; channel++)
            {
                unsigned char* pixelValuePtr = image.ptr(i) + (j * imageChannels) + channel;
                *pixelValuePtr = amount[channel][values[channel][index]];
            }
            index++;
        }
    }
}


int main(int argv, char** argc)
{
    Mat test = imread("F:\\resoursesVS\\OpCVFilt\\OpenCV.CMake\\8.jpg", IMREAD_UNCHANGED);
    Mat test2;
    //Mat test3;
    test.copyTo(test2);
    //test.copyTo(test3);
    //Mat1b a(test.rows, test.cols, 255);
    //Monochrome(test3);
    //SmoothingFilter(test2, 9);
    HistogramEqualization(test2);
    //GaussianFilter(test, 5);
    //sobel(test2);
    imshow("test", test);
    imshow("test2", test2);
    //imshow("test3", test3);
    waitKey();
}

