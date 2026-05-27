#include <apriltag_localization/apriltag_localization.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "apriltag_localization_node");

    ros::NodeHandle nh;

    ros::Rate rate(30);

    APRILTAG_LOCALIZATION tag_localization(&nh, rate);

    int counter = 0;
    while(ros::ok())
    {
        ros::spinOnce();
        tag_localization.getTrueRT();

        rate.sleep();
    }
}