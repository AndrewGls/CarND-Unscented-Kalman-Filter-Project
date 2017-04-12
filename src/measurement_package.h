#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_

#include "Eigen/Dense"

class MeasurementPackage
{
public:
    long long timestamp_;

    enum SensorType{
        LASER,
        RADAR
    } sensor_type_;

    Eigen::VectorXd raw_measurements_;
    
    bool IsLidar() const { return sensor_type_ == LASER; }
    bool IsRadar() const { return sensor_type_ == RADAR; }
};

#endif /* MEASUREMENT_PACKAGE_H_ */
