#include "utility.h"
#include <iostream>
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    std::cout << "ng1: "<< ng1.transpose() << std::endl;
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    // std::cout << "R0.FromTwoVectors(ng1, ng2): "<< R0 << std::endl;
    // double yaw = Utility::R2ypr(R0).x();
    // std::cout << "yaw(R0): "<< yaw << std::endl;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
