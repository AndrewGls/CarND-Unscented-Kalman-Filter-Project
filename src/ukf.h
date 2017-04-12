#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

    
  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;
    
  ///* time when the state is true, in us
  long long time_us_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF() {}

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
    void ProcessMeasurement(const MeasurementPackage& meas_package);

private:
    void Init(const MeasurementPackage& meas_package);
    
  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(const MeasurementPackage& meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(const MeasurementPackage& meas_package);
    
    // Calculates sigma points used in unscented transformation for approximation
    // of real distribution of non-lenear motion model.
    void CalcAugmentedSigmaPoints();
    
    // Predicts sigma points for non-linear motion model using augmented sigma points.
    void PredictSigmaPoints(double delta_t);

    // Predicts state (mean) and the state covariance matrix.
    void PredictCovariance();
    
    // Transform sigma points into measurement space
    MatrixXd TransformRadarSigmaPointsIntoMeasurementSpace();
    MatrixXd TransformLidarSigmaPointsIntoMeasurementSpace();
    
    // calculate predicted mean for sigma points.
    static VectorXd CalcPredictedMean(const MatrixXd& Xsig_pred, const VectorXd& weights);
        
private:
    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;
    
    // previous timestamp
    long long previous_timestamp_;
    
    // State dimension
    const int n_x_;
    
    // Augmented state dimension
    const int n_aug_;
    
    // number of sigma points
    const int n_sigma_;
    
    const int n_z_lidar_; // measurement dimension, radar can measure px and py
    const int n_z_radar_; // measurement dimension, radar can measure r, phi, and r_dot
    
    const int n_std_a_;     // position (row, col) of noise std_a in Predition covariance matrix.
    const int n_std_yawdd_; // position (row, col) of noise std_yawdd_ in Predition covariance matrix.
    
    // Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;
    
    // Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;
    
    // Sigma point spreading parameter
    const double lambda_;

    // Laser measurement noise covariance matrix
    MatrixXd R_laser_;
    
    // Radar measurement noise covariance matrix
    MatrixXd R_radar_;
    
    // Augmented sigma points matrix consists of augmented state vectors.
    MatrixXd Xsig_aug_;
};

#endif /* UKF_H */
